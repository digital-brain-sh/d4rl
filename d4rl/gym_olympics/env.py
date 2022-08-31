from .offline_env import OfflineEnv
import gym
import numpy as np
from gym import spaces
from .envs.rd_running import rd_running
from .envs.football import football
from .envs.table_hockey import table_hockey
from .envs.wrestling import wrestling


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

env_name_mapper = {
    'jidi-olympics-rd_running-expert-v0': rd_running,
    'jidi-olympics-table_hockey-expert-v0': table_hockey,
    'jidi-olympics-football-expert-v0': football,
    'jidi-olympics-wrestling-expert-v0': wrestling,
}




def post_process(obs):
    # batch 2 40 40
    obs = np.expand_dims(obs, axis=2)
    # batch 2 1 40 40
    obs = np.repeat(obs, repeats=3, axis=2)
    # batch 2 3 40 40
    return obs.astype(np.float32)

class OlympicsEnv(gym.Env):
    def __init__(self,
                game,
                **kwargs) -> None:
        super().__init__()
        env = env_name_mapper[game](max_episode_steps=kwargs.get('max_episode_steps'))
        self._env = env
        self.post_process_fn = post_process
        self.observation_space = env.observation_space
        self.action_space = spaces.Box(-1, 1, shape=(2, 2), dtype=np.float)
        self.observation_space = {
            'obs': spaces.Box(0, 255, shape=(2, 3, 40, 40), dtype=np.uint8),
            'energy': spaces.Box(0, 1000, shape=(2, ), dtype=np.float),
            'new_game': spaces.Box(False, True, shape=(1, ), dtype=np.bool),
        }

    def step(self, action):
        obs, reward, done, info = self._env.step(self.action_mapper(action))
        return {'obs': self.obs_wrapper(obs), 'energy': np.array(info['energy']), 'new_game': done}, reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return {'obs': self.obs_wrapper(obs), 'energy': np.array([1000., 1000.]), 'new_game': np.array(True)}

    def render(self):
        self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)
    
    def action_mapper(self, action):
        # action_f = [-100, 200]
        # action_theta = [-30, 30]
        # map action of [-1, 1] to [-100, 200] and [-30, 30]
        action[:, 0] = action[:, 0] * 150. + 50.
        action[:, 1] = action[:, 1] * 30.
        action = list(action)
        return action
    
    def obs_wrapper(self, obs):
        obs = np.expand_dims(obs, axis=1)
        obs = np.repeat(obs, repeats=3, axis=1)
        return obs.astype(np.float)


class OfflineOlympicsEnv(OlympicsEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        OlympicsEnv.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)

