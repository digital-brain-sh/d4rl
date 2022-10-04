from multiprocessing.connection import Connection
import warnings
import gym
import gc
import numpy as np

from multiprocessing import Process, Pipe
from gym import spaces
from d4rl.offline_env import OfflineEnv

from isaacgym import gymapi

import torch
from legged_gym import envs
from legged_gym.utils import task_registry, get_args
from legged_gym.envs import LeggedRobot


def parse_info_dict(info_dict: dict):
    res = {}
    for k, v in info_dict.items():
        if isinstance(v, dict):
            res[k] = parse_info_dict(v)
        elif isinstance(v, torch.Tensor):
            res[k] = v.cpu().numpy().squeeze()
        else:
            raise TypeError(f"unexpected type: {type(v)} for key={k}")
    return res


def get_env_args(name):
    if name == "anymal_c_rough":
        from legged_gym.envs.anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg
        return AnymalCRoughCfg()
    elif name == "anymal_c_flat":
        from legged_gym.envs.anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg
        return AnymalCFlatCfg()
    elif name == "anymal_b":
        from legged_gym.envs.anymal_b.anymal_b_config import AnymalBRoughCfg
        return AnymalBRoughCfg()
    elif name == "a1":
        from legged_gym.envs.a1.a1_config import A1RoughCfg
        return A1RoughCfg()


def task_running(main, conn: Connection, task_name: str, mesh_type: str = "trimesh"):
    main.close()
    leg_args = get_args()
    leg_args.task = task_name
    leg_args.headless = True
    leg_args.sim_device = "cpu"
    leg_args.sim_device_type = "cpu"
    leg_args.pipeline = 'cpu'
    leg_args.use_gpu = False
    leg_args.use_gpu_pipeline = False
    leg_args.rl_device = "cpu"
    leg_args.device = "cpu"
    leg_args.num_envs = 1
    env_args = get_env_args(task_name)
    env_args.seed = 1
    env_args.terrain.mesh_type = mesh_type

    # leg_args.physics_engine = gymapi.SIM_FLEX
    env, task_config = task_registry.make_env(task_name, leg_args, env_args)
    while True:
        # 0: operation, 1: data
        msg = conn.recv()
        if msg[0] == "get_task_config":
            conn.send(task_config)
        elif msg[0] == "reset":
            obs, privileged_obs = env.reset()
            obs = obs.cpu().numpy().squeeze()
            if privileged_obs is not None:
                privileged_obs = privileged_obs.cpu().numpy().squeeze()
            conn.send((obs, privileged_obs))
        elif msg[0] == "step":
            action = msg[1]
            action = torch.FloatTensor([action], device="cpu")
            obs, privileged_obs, rew, done, info = env.step(action)
            obs = obs.cpu().numpy().squeeze()
            rew = rew.cpu().numpy().squeeze()
            done = done.cpu().numpy().squeeze()
            if privileged_obs is not None:
                privileged_obs = privileged_obs.cpu().numpy().squeeze()
            # convert info tensor dict to numpy dict
            info = parse_info_dict(info)
            conn.send((obs, privileged_obs, rew, done, info))
        elif msg[0] == "close":
            del env
            conn.send(200)
            break


class RoboticsEnv(gym.Env):

    def __init__(self, task_name: str, config: str = "default") -> None:
        super().__init__()

        leg_args = get_args()
        leg_args.task = task_name
        leg_args.headless = True
        leg_args.sim_device = "cpu"
        leg_args.sim_device_type = "cpu"
        leg_args.pipeline = 'cpu'
        leg_args.use_gpu = False
        leg_args.use_gpu_pipeline = False
        leg_args.rl_device = "cpu"
        leg_args.device = "cpu"
        leg_args.num_envs = 1

        if config == "default":
            env_cfg = None
        else:
            raise NotImplementedError

        self.env, task_config = task_registry.make_env(task_name, leg_args, env_cfg)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(task_config.env.num_observations,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(task_config.env.num_actions,), dtype=np.float32)

    def step(self, action):
        action = torch.FloatTensor(action, device="cpu")
        obs, privileged_obs, rew, done, info = self.env.step(action)
        obs = obs.cpu().numpy().squeeze()
        rew = rew.cpu().numpy().squeeze()
        done = done.cpu().numpy().squeeze()
        if privileged_obs is not None:
            privileged_obs = privileged_obs.cpu().numpy().squeeze()
        # convert info tensor dict to numpy dict
        info = parse_info_dict(info)
        return obs, rew, done, info

    def reset(self):
        obs, privileged_obs = self.env.reset()
        obs = obs.cpu().numpy().squeeze()
        if privileged_obs is not None:
            privileged_obs = privileged_obs.cpu().numpy().squeeze()
        return obs


class SubPorcRoboticsEnv(gym.Env):

    def __init__(self, task_name: str, mesh_type) -> None:
        super().__init__()

        conn1, conn2 = Pipe()
        if mesh_type == "None":
            mesh_type = None
        self.env_process = Process(target=task_running, args=(conn1, conn2, task_name, mesh_type))
        self.env_process.start()

        conn1.send(["get_task_config"])
        task_config = conn1.recv()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(task_config.env.num_observations,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(task_config.env.num_actions,), dtype=np.float32)
        self.conn1 = conn1
        self.conn2 = conn2

    def __del__(self):
        if not self.conn1.closed:
            self.conn1.send(["close"])
            code = self.conn1.recv()
            self.conn1.close()
            self.conn2.close()
            self.env_process.join()

        del self.conn1
        del self.conn2
        del self.env_process

    def step(self, action):
        # convert action to tensor
        self.conn1.send(["step", action])
        obs, privileged_obs, rew, done, info = self.conn1.recv()
        return obs, rew, done, info

    def reset(self):
        self.conn1.send(["reset"])
        obs, privileged_obs = self.conn1.recv()
        return obs

    def close(self) -> None:
        if not self.conn1.closed:
            self.conn1.send(["close"])
            code = self.conn1.recv()
            self.conn1.close()
            self.conn2.close()


class OfflineRobotics(RoboticsEnv, OfflineEnv):

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, deprecated=False, deprecation_message=None, **kwargs):
        RoboticsEnv.__init__(self, kwargs["task_name"], kwargs["config"])
        OfflineEnv.__init__(self, dataset_url, ref_max_score, ref_min_score, deprecated, deprecation_message, **kwargs)