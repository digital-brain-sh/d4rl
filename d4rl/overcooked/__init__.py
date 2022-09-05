from gym.envs.registration import register
from d4rl import infos


DATASET_URLS = {}
ALL_ENVS = []

TODO_LAYOUTS = [
    'bottleneck',
    'centre_objects',
    'coordination_ring',
    'corridor',
    'counter_circuit',
    'five_by_five',
    'forced_coordination_tomato',
    'large_room',
    'long_cook_time',
    'pipeline',
    'schelling_s',
    'small_corridor',

]

DONE_LAYOUTS = [
    'asymmetric_advantages_tomato',
    'asymmetric_advantages',
    'centre_pots',
    'cramped_room_o_3orders',
    'cramped_room',
    'forced_coordination',
    'inverse_marshmallow_experiment',
    'marshmallow_experiment',
    'soup_coordination',
    'unident',
    'you_shall_not_pass',
]

ALL_LAYOUTS = DONE_LAYOUTS + TODO_LAYOUTS

for game in ALL_LAYOUTS:
    env_name = game + '-expert-v0'
    ALL_ENVS.append(env_name)
    register(
        id=env_name,
        entry_point='d4rl.overcooked.envs:OfflineOvercookedEnv',
        kwargs={
            'game': game,
            'horizon': 400,
            'ref_min_score': infos.REF_MIN_SCORE.get(env_name, None),
            'ref_max_score': infos.REF_MAX_SCORE.get(env_name, None),
        }
    )
