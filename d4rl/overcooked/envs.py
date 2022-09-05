import gym
from .offline_env import OfflineEnv
from gym import spaces
import numpy as np
from dataclasses import dataclass
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

OvercookedAction = Action.ALL_ACTIONS

class GatoOvercookedObsWrapper(gym.ObservationWrapper):
    """Wrap observation of Overcooked games for Gato pretraining
    1. change 64 * 64 * 3 image to 3 * 64 * 64
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(
            env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]),
                                            dtype=np.uint8)

    def observation(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.float32)


def post_process(obs):
    # k * 2 * ? * ? * 26
    obs = np.reshape(obs, (obs.shape[0], obs.shape[1], -1))
    return obs.astype(np.float32)

class OvercookedEnvWrapper(gym.Env):
    def __init__(self,
                 game,
                 **kwargs):
        mdp_params = {"layout_name": game}
        env_params = {"horizon": kwargs.get('horizon', 400)}
        mdp = OvercookedGridworld.from_layout_name(**mdp_params)
        assert type(mdp) == OvercookedGridworld, "mdp must be a OvercookedGridworld object"
        mdp_fn = lambda _ignored: mdp
        self._env = OvercookedEnv(mdp_fn, **env_params)
        self.post_process_fn = post_process
        dummy_state = self._env.mdp.get_standard_start_state()
        self.featurize_fn = lambda state: self._env.lossless_state_encoding_mdp(state)
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        high = np.reshape(high, [2, -1])
        low = np.reshape(low, [2, -1])
        self.observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

    def step(self, action):
        joint_action = (OvercookedAction[action[0]], OvercookedAction[action[1]])
        (next_state, timestep_sparse_reward, done, env_info) = self._env.step(joint_action, display_phi=False)
        return self.__obs_wrapper__(next_state), timestep_sparse_reward, done, env_info

    def reset(self, **kwargs):
        self._env.reset()
        return self.__obs_wrapper__(self._env.state)

    def __obs_wrapper__(self, state):
        obs = np.array(self.featurize_fn(state))
        obs = np.reshape(obs, [2, -1])
        return obs

    def render(self):
        pass

    def seed(self, seed=None):
        pass

from src.data.input_specs import RLTaskInput

@dataclass
class OvercookedInput(RLTaskInput):
    pass

class OfflineOvercookedEnv(OvercookedEnvWrapper, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        OvercookedEnvWrapper.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)
    
    def build_task_input(self, *args, **kwargs):
        return OvercookedInput(*args, **kwargs)
