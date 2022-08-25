import gym
from .offline_env import OfflineEnv
from .dmlab_env import DmLab
from gym import spaces
import numpy as np
from dataclasses import dataclass


# from src.data.input_specs import RLTaskInput
#
# @dataclass
# class DMLABInput(RLTaskInput):
#     pass

class OfflineDMLABEnv(DmLab, OfflineEnv):
    def __init__(self, **kwargs):
        game = kwargs['game']
        del kwargs['game']
        DmLab.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)
    #
    # def build_task_input(self, *args, **kwargs):
    #     return DMLABInput(*args, **kwargs)
