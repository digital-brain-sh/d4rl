import gym
from .offline_env import OfflineEnv
from gym import spaces
import numpy as np
import copy


class GatoProcgenObsWrapper(gym.ObservationWrapper):
    """Wrap observation of Procgen games for Gato pretraining
    1. change 64 * 64 * 3 image to 3 * 64 * 64
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(
            env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]),
                                            dtype=np.uint8)

    def observation(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        return obs


class ProcgenEnv(gym.Env):
    def __init__(self,
                 game,
                 **kwargs):
        self.distribution_mode = kwargs.get('distribution_mode')
        self.start_level = kwargs.get('start_level')
        self.num_levels = kwargs.get('num_levels')
        env = gym.make(game, start_level=self.start_level, num_levels=self.num_levels,
                       distribution_mode=self.distribution_mode)
        env = GatoProcgenObsWrapper(env)
        self._env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def render(self):
        self._env.render()

    def seed(self, seed=None):
        super().seed(seed)
        self._env.seed(seed)


class OfflineProcgenEnv(ProcgenEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        ProcgenEnv.__init__(self, game=game)
        OfflineEnv.__init__(self, game=game, **kwargs)
