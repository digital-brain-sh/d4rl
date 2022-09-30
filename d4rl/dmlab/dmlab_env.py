# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepMind Lab Gym wrapper."""

import hashlib
import os
import gym
import numpy as np
import deepmind_lab
import shutil
from gym import spaces

DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),  # Fire.
)

NEW_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-10, 0, 0, 0, 0, 0, 0),  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0),   # Small Look Right
    (-60, 0, 0, 0, 0, 0, 0),  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0),   # Large Look Right
    (0, 10, 0, 0, 0, 0, 0),   # Look Down
    (0, -10, 0, 0, 0, 0, 0),  # Look Up
    (-10, 0, 0, 1, 0, 0, 0),  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0),   # Forward + Small Look Right
    (-60, 0, 0, 1, 0, 0, 0),  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0),   # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)

NEW_ACTION_DECODER = {
    (0, 0, 0, 1, 0, 0, 0): 0,
    (0, 0, 0, -1, 0, 0, 0): 1,  # Backward
    (0, 0, -1, 0, 0, 0, 0): 2,  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0): 3,  # Strafe Right
    (-10, 0, 0, 0, 0, 0, 0): 4,  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0): 5,   # Small Look Right
    (-60, 0, 0, 0, 0, 0, 0): 6,  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0): 7,   # Large Look Right
    (0, 10, 0, 0, 0, 0, 0): 8,   # Look Down
    (0, -10, 0, 0, 0, 0, 0): 9,  # Look Up
    (-10, 0, 0, 1, 0, 0, 0): 10,  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0): 11,   # Forward + Small Look Right
    (-60, 0, 0, 1, 0, 0, 0): 12,  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0): 13,   # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0): 14,    # Fire.
}

NEW_ACTION_DECODER1 = {
    (0, 0, 0, 1, 0, 0, 0): 0,
    (0, 0, 0, 255, 0, 0, 0): 1,  # Backward
    (0, 0, 255, 0, 0, 0, 0): 2,  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0): 3,  # Strafe Right
    (246, 0, 0, 0, 0, 0, 0): 4,  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0): 5,   # Small Look Right
    (196, 0, 0, 0, 0, 0, 0): 6,  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0): 7,   # Large Look Right
    (0, 10, 0, 0, 0, 0, 0): 8,   # Look Down
    (0, 246, 0, 0, 0, 0, 0): 9,  # Look Up
    (246, 0, 0, 1, 0, 0, 0): 10,  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0): 11,   # Forward + Small Look Right
    (196, 0, 0, 1, 0, 0, 0): 12,  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0): 13,   # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0): 14,    # Fire.
}


ACTION_DECODER = {
    (0, 0, 0, 1, 0, 0, 0): 0,
    (0, 0, 0, -1, 0, 0, 0): 1, 
    (0, 0, -1, 0, 0, 0, 0): 2, 
    (0, 0, 1, 0, 0, 0, 0): 3, 
    (-20, 0, 0, 0, 0, 0, 0): 4, 
    (20, 0, 0, 0, 0, 0, 0): 5, 
    (-20, 0, 0, 1, 0, 0, 0): 6, 
    (20, 0, 0, 1, 0, 0, 0): 7, 
    (0, 0, 0, 0, 1, 0, 0): 8, 
}


class LevelCache(object):

    def __init__(self, cache_dir='../dmlab_level_cache'):
        self._cache_dir = cache_dir

    def get_path(self, key):
        key = hashlib.md5(key.encode('utf-8')).hexdigest()
        dir_, filename = key[:5], key[5:]
        return os.path.join(self._cache_dir, dir_, filename)

    def fetch(self, key, pk3_path):
        path = self.get_path(key)
        if os.path.isfile(path):
            shutil.copyfile(path, pk3_path)
            return True
        return False

    def write(self, key, pk3_path):
        path = self.get_path(key)
        if not os.path.isfile(path):
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            try:
                shutil.copyfile(pk3_path, path)
            except:
                print('key: ', key)
                print('pk3_path: ', pk3_path)
                print('path: ', path)
                raise

def post_process(obs):
    obs = np.transpose(obs, (0, 3, 1, 2))
    obs = np.pad(obs, ((0, 0), (0, 0), (4, 4), (0, 0)), 'constant').astype(np.float32)
    return obs

def new_post_process(obs):
    obs = np.pad(obs, ((0, 0), (0, 0), (4, 4), (0, 0)), 'constant').astype(np.float32)
    return obs  

def action_mapper(actions):
    if len(actions.shape) == 1:
        return np.array(ACTION_DECODER[tuple(actions)])
    else:
        decoded_action = []
        for item in actions[:]:
            decoded_action.append(ACTION_DECODER[tuple(item)])
        decoded_action = np.array(decoded_action, dtype=np.int64)
        return decoded_action

def gen_action_mapper(actions):
    if len(actions.shape) == 1:
        return np.array(NEW_ACTION_DECODER1[tuple(actions)])
    else:
        decoded_action = []
        for item in actions[:]:
            decoded_action.append(NEW_ACTION_DECODER1[tuple(item)])
        decoded_action = np.array(decoded_action, dtype=np.int64)
        return decoded_action


class GatoDMLABObsWrapper(gym.ObservationWrapper):
    """Wrap observation of DMLAB games for Gato pretraining
    1. change 72 * 96 * 3 image to 3 * 80 * 96
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(3, 80, 96), dtype=np.uint8)

    def observation(self, obs):
        print(obs.shape)
        obs = np.transpose(obs, (2, 0, 1))
        # pad 3 * 72 * 96 to 3 * 80 * 96
        obs = np.pad(obs, ((0, 0), (4, 4), (0, 0)), 'constant')
        print(obs.shape)
        return obs

class DmLab(gym.Env):
    """DeepMind Lab wrapper."""

    def __init__(self, game, **kwargs):
        self.gen_id = kwargs.get('gen_id')
        if self.gen_id == 0:
            self.post_process_fn = post_process
            self.action_mapper = action_mapper
            self._action_set = DEFAULT_ACTION_SET
        elif self.gen_id == 1 or self.gen_id == -1:
            self.post_process_fn = new_post_process
            self.action_mapper = gen_action_mapper
            self._action_set = NEW_ACTION_SET
        config = {}
        if 'test' in game:
            config['allowHoldOutLevels'] = 'true'
            # Mixer seed for evalution, see
            # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
            config['mixerSeed'] = 0x600D5EED
        self.level = game
        self.extra_input = kwargs.get('extra_input', False)
        config['logLevel'] = 'WARN'
        config['width'] = 96
        config['height'] = 72
        config['datasetPath'] = '/raid/brady_konkle_oliva2008'
        self._num_action_repeats = kwargs.get('num_action_repeats', None)
        self._random_state = np.random.RandomState(seed=kwargs.get('seed', 0))
        self._env = deepmind_lab.Lab(
            level=self.level,
            observations=['RGB_INTERLEAVED', 'INSTR'],
            level_cache=LevelCache(),
            config={k: str(v) for k, v in config.items()},
        )
        # self._env = deepmind_lab.Lab(
        #     level=self.level,
        #     observations=['RGB_INTERLEAVED', 'INSTR'],
        #     config={k: str(v) for k, v in config.items()},
        # )
        
        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, 80, 96),
            dtype=np.uint8)

    def _obs_wrapper(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        # pad 3 * 72 * 96 to 3 * 80 * 96
        obs = np.pad(obs, ((0, 0), (4, 4), (0, 0)), 'constant').astype(np.float32)
        return obs

    def _observation(self):
        if self.extra_input:
            obs = self._env.observations()
            obs['RGB_INTERLEAVED'] = self._obs_wrapper(obs['RGB_INTERLEAVED'])
            return [obs[k] for k in obs.keys()]
        else:
            return self._obs_wrapper(self._env.observations()['RGB_INTERLEAVED'])

    def reset(self, **kwargs):
        self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
        return self._observation()

    def step(self, action):
        raw_action = np.array(self._action_set[action], np.intc)
        # try:
        reward = self._env.step(raw_action, num_steps=self._num_action_repeats)
        # except Exception as e:
        #     print(e.__class__.__name__, e)
        #     observation = None
        #     instruction = None
        #     reward = np.array(0.)
        #     done = np.array(True)
        #     return [observation, instruction], reward, done, {}
        done = not self._env.is_running()
        if done:
            observation = np.zeros([3, 80, 96], dtype=np.float32)
            # observation = None
        else:
            observation = self._observation()
        return observation, reward, done, {}

    def seed(self, seed):
        self._random_state = np.random.RandomState(seed=seed)

    def close(self):
        self._env.close()
