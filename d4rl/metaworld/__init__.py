import os
import glob
from gym.envs.registration import register, registry
"""Proposal for a simple, understandable MetaWorld API."""
import abc
import pickle
from collections import OrderedDict
from typing import List, NamedTuple, Type

import d4rl.metaworld.core.mujoco.env_dict as _env_dict
import numpy as np

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv:
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """
    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_name is different from the current task.

        """


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args['args']) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args['kwargs'].copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
    if seed is not None:
        np.random.set_state(st0)
    return tasks


def _ml1_env_names():
    tasks = list(_env_dict.ML1_V2['train'])
    assert len(tasks) == 50
    return tasks


class ML1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        _ML_OVERRIDE,
                                        seed=seed)
        self._test_tasks = _make_tasks(
            self._test_classes, {env_name: args_kwargs},
            _ML_OVERRIDE,
            seed=(seed + 1 if seed is not None else seed))


class MT1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes,
                                        {env_name: args_kwargs},
                                        _MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []


class ML10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V2['train']
        self._test_classes = _env_dict.ML10_V2['test']
        train_kwargs = _env_dict.ml10_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _ML_OVERRIDE,
                                        seed=seed)
        test_kwargs = _env_dict.ml10_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs,
                                       _ML_OVERRIDE,
                                       seed=seed)


class ML45(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V2['train']
        self._test_classes = _env_dict.ML45_V2['test']
        train_kwargs = _env_dict.ml45_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _ML_OVERRIDE,
                                        seed=seed)
        test_kwargs = _env_dict.ml45_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs,
                                       _ML_OVERRIDE,
                                       seed=seed)


class MT10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []


class MT50(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed)
        self._test_tasks = []

# __all__ = ["ML1", "MT1", "ML10", "MT10", "ML45", "MT50"]


REF_MAX_SCORE={
    'bin-picking-v2': 4292.8,
    'button-press-topdown-v2': 3888.4,
    'button-press-topdown-wall-v2': 3881.9,
    'button-press-v2': 3623.3,
    'button-press-wall-v2': 3675.1,
    'coffee-button-v2': 4261.5,
    'coffee-pull-v2': 4201.5,
    'coffee-push-v2': 4184.8,
    'dial-turn-v2': 4658.2,
    'door-close-v2': 4540.0,
    'door-lock-v2': 3913.5,
    'door-open-v2': 4561.0,
    'door-unlock-v2': 4625.9,
    'drawer-close-v2': 4871.3,
    'drawer-open-v2': 4224.6,
    'faucet-close-v2': 4757.6,
    'faucet-open-v2': 4774.3,
    'hammer-v2': 4613.9,
    'hand-insert-v2': 4544.1,
    'handle-press-side-v2': 4844.2,
    'handle-press-v2': 4867.2,
    'handle-pull-side-v2': 4635.5,
    'handle-pull-v2': 4492.6,
    'peg-insert-side-v2': 4605.7,
    'peg-unplug-side-v2': 4464.6,
    'pick-place-v2': 4428.6,
    'plate-slide-back-side-v2': 4788.9,
    'plate-slide-back-v2': 4777.9,
    'plate-slide-side-v2': 4683.2,
    'plate-slide-v2': 4674.1,
    'push-back-v2': 4237.2,
    'push-v2': 4750.8,
    'push-wall-v2': 4676.8,
    'reach-v2': 4863.4,
    'reach-wall-v2': 4806.4,
    'soccer-v2': 4463.7,
    'stick-pull-v2': 4228.8,
    'sweep-into-v2': 4631.5,
    'sweep-v2': 4487.1,
    'window-close-v2': 4563.4,
    'window-open-v2': 4406.4,
    
}
# env_names = glob.glob('/nfs/dgx08/home/lcy/metaworld/garage/datasets/*.hdf5')

# env_names = [os.path.basename(env_name)[:-8] for env_name in env_names]

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for ref_key, ref_max_score in REF_MAX_SCORE.items():
    agent = ref_key[:-3]
    offline_env_name = 'mw-%s-expert-v0' % agent
    if offline_env_name in registry.env_specs:
            continue
    register(
        id=offline_env_name,
        entry_point='d4rl.metaworld.envs:OfflineMWRlEnv',
        kwargs={
            'game': f'{agent}-v2',
            'ref_min_score': 0,
            'ref_max_score': REF_MAX_SCORE[f'{agent}-v2'],
            'dataset_url': f'/nfs/dgx08/home/lcy/metaworld/garage/datasets/{agent}-v2.hdf5'
        }
    )

# ALL_ENVS = env_names

if __name__ == "__main__":
    import gym, time

    for key in REF_MAX_SCORE.keys():
        game = "mw-" + key[:-3] + "-expert-v0"
        env = gym.make(game)
        print(f"* game {game} created")
        done = False
        n_step = 0
        start = time.time()
        obs = env.reset()
        dataset = env.get_dataset()
        while not done:
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            n_step += 1
        time_clsp = time.time() - start
        print(f"* game {game} ended for {n_step} step(s), time consump: {time_clsp}")