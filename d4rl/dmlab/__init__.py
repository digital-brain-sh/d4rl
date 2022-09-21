from gym.envs.registration import register
from .games import *
from d4rl import infos
import os

DATASET_URLS = {}
ALL_ENVS = []

ALL_LEVELS = DMLAB_30 + psychlab + psychlab_memory + psychlab_visuospatial + fast_mapping + others

GENERATED_LEVELS = gen_explore_goal_locations + \
    gen_explore_object_locations + gen_explore_object_rewards + \
    gen_explore_obstructed_goals + gen_lasertag

NEW_GEN_LEVELS = gen_explore_goal_locations + \
    gen_explore_object_locations + gen_explore_object_rewards + \
    gen_explore_obstructed_goals + gen_lasertag


dataset_pos = {
    'dgx05': ['explore_goal_locations_large',
              'explore_object_locations_large',
              'explore_object_rewards_few',
              'explore_obstructed_goals_large',
              'nav_maze_static_01',
              'skymaze_irreversible_path_varied',
              'explore_goal_locations_small',
              'explore_object_locations_small',
              'explore_object_rewards_many',
              'explore_obstructed_goals_small',
              'skymaze_irreversible_path_hard',
              ],
    'dgx08': ['psychlab_arbitrary_visuomotor_mapping',
              'rooms_collect_good_objects_train',
              'rooms_keys_doors_puzzle',
              'rooms_watermaze',
              'psychlab_continuous_recognition',
              'rooms_exploit_deferred_effects_train',
              'rooms_select_nonmatching_object',
              ],
    'dgx09': ['natlab_fixed_large_map',
              'natlab_varying_map_randomized',
              'natlab_varying_map_regrowth',
              'nav_maze_static_02',
              'nav_maze_static_03',
              'seekavoid_arena_01',
              'stairway_to_melon',
              ]
}

for game in ALL_LEVELS:
    cur_game = None
    for taskset in tasksets.items():
        if game in taskset[1]:
            cur_game = os.path.join(tasksets_path[taskset[0]], game)
    position = ''
    for pos in dataset_pos.keys():
        if game in dataset_pos[pos]:
            position = pos
            break
    env_name = '%s-expert-v0' % game
    ALL_ENVS.append(env_name)
    register(
        id=env_name,
        entry_point='d4rl.dmlab.envs:OfflineDMLABEnv',
        kwargs={
            'game': cur_game,
            'gen_id': 0,
            'dataset_name': game,
            'extra_input': False if game not in languages else True,
            'ref_min_score': infos.REF_MIN_SCORE.get(env_name, None),
            'ref_max_score': infos.REF_MAX_SCORE.get(env_name, None),
            'is_test': False,
            'task_group': 'dmlab',
            'dataset_pos': position,
            'cache_path': None,
            'seed': 0,
            'num_action_repeats': 4,
        }
    )

for game in NEW_GEN_LEVELS:
    cur_game = None
    for taskset in tasksets.items():
        if game in taskset[1]:
            cur_game = os.path.join('new_gen_levels', tasksets_path[taskset[0]] + '_new', game)
    game = 'new_' + game
    position = ''
    for pos in dataset_pos.keys():
        if game in dataset_pos[pos]:
            position = pos
            break
    env_name = '%s-expert-v0' % game
    ALL_ENVS.append(env_name)
    register(
        id=env_name,
        entry_point='d4rl.dmlab.envs:OfflineDMLABEnv',
        kwargs={
            'game': cur_game,
            'gen_id': 1,
            'dataset_name': game,
            'extra_input': False if game not in languages else True,
            'ref_min_score': infos.REF_MIN_SCORE.get(env_name, None),
            'ref_max_score': infos.REF_MAX_SCORE.get(env_name, None),
            'is_test': False,
            'task_group': 'dmlab',
            'dataset_pos': position,
            'cache_path': None,
            'seed': 0,
            'num_action_repeats': 4,
        }
    )

for game in GENERATED_LEVELS:
    cur_game = None
    for taskset in tasksets.items():
        if game in taskset[1]:
            cur_game = os.path.join('old_gen_levels', tasksets_path[taskset[0]], game)
    position = ''
    for pos in dataset_pos.keys():
        if game in dataset_pos[pos]:
            position = pos
            break
    env_name = '%s-expert-v0' % game
    ALL_ENVS.append(env_name)
    register(
        id=env_name,
        entry_point='d4rl.dmlab.envs:OfflineDMLABEnv',
        kwargs={
            'game': cur_game,
            'gen_id': -1,
            'dataset_name': game,
            'extra_input': False if game not in languages else True,
            'ref_min_score': infos.REF_MIN_SCORE.get(env_name, None),
            'ref_max_score': infos.REF_MAX_SCORE.get(env_name, None),
            'is_test': False,
            'task_group': 'dmlab',
            'dataset_pos': position,
            'cache_path': None,
            'seed': 0,
            'num_action_repeats': 4,
        }
    )