from array import array
from .offline_env import OfflineEnv
import gym
import numpy as np
from gym import spaces
from .envs.olympics_env.olympics_integrated import OlympicsIntegrated
import os
import json


class GatoOlympicsObsWrapper(gym.ObservationWrapper):
    """Wrap observation of Procgen games for Gato pretraining
    1. change 64 * 64 * 3 image to 3 * 64 * 64
    """

    def __init__(self, env):
        super().__init__(env)
        

    def observation(self, obs):
        obs = np.expand_dims(obs, axis=1)
        obs = np.repeat(obs, repeats=3, axis=1)
        return obs

# olympics running env with the same map, map id can be specified as
# gym.make('gym_olympics:running-v0', map_id=1)

# olympics running env with map changing randomly
# gym.make('gym_olympics:rd_running-v0')

# gym.make('gym_olympics:table_hockey-v0')

# gym.make('gym_olympics:football-v0')

# gym.make('gym_olympics:wrestling-v0')




# def post_process(obs):
#     # batch 2 1602
#     obs = np.expand_dims(obs, axis=2)
#     # batch 2 1 40 40
#     obs = np.repeat(obs, repeats=3, axis=2)
#     # batch 2 3 40 40
#     return obs.astype(np.float32)

class OlympicsEnv(gym.Env):
    def __init__(self,
                game,
                **kwargs) -> None:
        super().__init__()
        config_path = os.path.join(os.path.dirname(__file__), 'envs', 'olympics_env', 'config.json')
        with open(config_path) as f:
            conf = json.load(f)['olympics-integrated']
        env = OlympicsIntegrated(conf, game)
        self._env = env
        # self.post_process_fn = post_process
        self.action_space = spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)
        self.observation_space = spaces.Box(0, 20., shape=(2, 1602), dtype=np.float32)
        # self.observation_space = {
        #     'obs': spaces.Box(0, 255, shape=(2, 3, 40, 40), dtype=np.uint8),
        #     'energy': spaces.Box(0, 1000, shape=(2, ), dtype=np.float),
        #     'new_game': spaces.Box(False, True, shape=(1, ), dtype=np.bool),
        # }
        self.new_game_mapper = {'NEW GAME': 1, '': 0}

    def step(self, action):
        obs, reward, done, info_before, info_after = self._env.step(self.action_mapper(action))
        return self.obs_wrapper(obs), reward, done, {}

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return self.obs_wrapper(obs)

    def render(self):
        self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)

    def obs_wrapper(self, obs):
        obs = [obs[0]['obs'], obs[1]['obs']]
        array_obs = [obs[0]['agent_obs'].flatten(), obs[0]['agent_obs'].flatten()]
        game_mode = [self.new_game_mapper[obs[0]['game_mode']], self.new_game_mapper[obs[1]['game_mode']]]
        energy = [obs[0]['energy'] / 1000, obs[1]['energy'] / 1000]
        obs = np.array([np.hstack([array_obs[0], game_mode[0], energy[0]]), np.hstack([array_obs[1], game_mode[1], energy[1]])])
        # print(obs.shape)
        return obs
    
    def action_mapper(self, action):
        # action_f = [-100, 200]
        # action_theta = [-30, 30]
        # map action of [-1, 1] to [-100, 200] and [-30, 30]
        action[:, 0] = action[:, 0] * 150. + 50.
        action[:, 1] = action[:, 1] * 30.
        action = [[[action[0][0]], [action[0][1]]], [[action[1][0]], [action[1][1]]]]
        # print(action)
        return action


from src.data.input_specs import RLTaskInput

@dataclass
class OlympicsInput(RLTaskInput):
    pass


class OfflineOlympicsEnv(OlympicsEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        OlympicsEnv.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)

    def build_task_input(self, *args, **kwargs):
        return OlympicsInput(*args, **kwargs)
