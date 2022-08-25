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

"""Contains list of DeepMind Lab games, human/random scores and utilities."""

import collections
import numpy as np


GAME_MAPPING = collections.OrderedDict([
    ('rooms_collect_good_objects_train', 'rooms_collect_good_objects_test'),
    ('rooms_exploit_deferred_effects_train',
     'rooms_exploit_deferred_effects_test'),
    ('rooms_select_nonmatching_object', 'rooms_select_nonmatching_object'),
    ('rooms_watermaze', 'rooms_watermaze'),
    ('rooms_keys_doors_puzzle', 'rooms_keys_doors_puzzle'),
    ('language_select_described_object', 'language_select_described_object'),
    ('language_select_located_object', 'language_select_located_object'),
    ('language_execute_random_task', 'language_execute_random_task'),
    ('language_answer_quantitative_question',
     'language_answer_quantitative_question'),
    ('lasertag_one_opponent_small', 'lasertag_one_opponent_small'),
    ('lasertag_three_opponents_small', 'lasertag_three_opponents_small'),
    ('lasertag_one_opponent_large', 'lasertag_one_opponent_large'),
    ('lasertag_three_opponents_large', 'lasertag_three_opponents_large'),
    ('natlab_fixed_large_map', 'natlab_fixed_large_map'),
    ('natlab_varying_map_regrowth', 'natlab_varying_map_regrowth'),
    ('natlab_varying_map_randomized', 'natlab_varying_map_randomized'),
    ('skymaze_irreversible_path_hard', 'skymaze_irreversible_path_hard'),
    ('skymaze_irreversible_path_varied', 'skymaze_irreversible_path_varied'),
    ('psychlab_arbitrary_visuomotor_mapping',
     'psychlab_arbitrary_visuomotor_mapping'),
    ('psychlab_continuous_recognition', 'psychlab_continuous_recognition'),
    ('psychlab_sequential_comparison', 'psychlab_sequential_comparison'),
    ('psychlab_visual_search', 'psychlab_visual_search'),
    ('explore_object_locations_small', 'explore_object_locations_small'),
    ('explore_object_locations_large', 'explore_object_locations_large'),
    ('explore_obstructed_goals_small', 'explore_obstructed_goals_small'),
    ('explore_obstructed_goals_large', 'explore_obstructed_goals_large'),
    ('explore_goal_locations_small', 'explore_goal_locations_small'),
    ('explore_goal_locations_large', 'explore_goal_locations_large'),
    ('explore_object_rewards_few', 'explore_object_rewards_few'),
    ('explore_object_rewards_many', 'explore_object_rewards_many'),
])

languages = [
    'language_select_described_object',
    'language_select_located_object',
    'language_execute_random_task',
    'language_answer_quantitative_question',
]

ALL_GAMES = frozenset([
    'rooms_collect_good_objects_train',
    'rooms_collect_good_objects_test',
    'rooms_exploit_deferred_effects_train',
    'rooms_exploit_deferred_effects_test',
    'rooms_select_nonmatching_object',
    'rooms_watermaze',
    'rooms_keys_doors_puzzle',
    'language_select_described_object',
    'language_select_located_object',
    'language_execute_random_task',
    'language_answer_quantitative_question',
    'lasertag_one_opponent_small',
    'lasertag_three_opponents_small',
    'lasertag_one_opponent_large',
    'lasertag_three_opponents_large',
    'natlab_fixed_large_map',
    'natlab_varying_map_regrowth',
    'natlab_varying_map_randomized',
    'skymaze_irreversible_path_hard',
    'skymaze_irreversible_path_varied',
    'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition',
    'psychlab_sequential_comparison',
    'psychlab_visual_search',
    'explore_object_locations_small',
    'explore_object_locations_large',
    'explore_obstructed_goals_small',
    'explore_obstructed_goals_large',
    'explore_goal_locations_small',
    'explore_goal_locations_large',
    'explore_object_rewards_few',
    'explore_object_rewards_many',
])

DMLAB_30 = [
    'rooms_collect_good_objects_train',
    'rooms_exploit_deferred_effects_train',
    'rooms_select_nonmatching_object',
    'rooms_exploit_deferred_effects_test',
    'rooms_collect_good_objects_test',
    'rooms_watermaze',
    'rooms_keys_doors_puzzle',
    'language_select_described_object',
    'language_select_located_object',
    'language_execute_random_task',
    'language_answer_quantitative_question',
    'lasertag_one_opponent_small',
    'lasertag_three_opponents_small',
    'lasertag_one_opponent_large',
    'lasertag_three_opponents_large',
    'natlab_fixed_large_map',
    'natlab_varying_map_regrowth',
    'natlab_varying_map_randomized',
    'skymaze_irreversible_path_hard',
    'skymaze_irreversible_path_varied',
    'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition',
    'psychlab_sequential_comparison',
    'psychlab_visual_search',
    'explore_object_locations_small',
    'explore_object_locations_large',
    'explore_obstructed_goals_small',
    'explore_obstructed_goals_large',
    'explore_goal_locations_small',
    'explore_goal_locations_large',
    'explore_object_rewards_few',
    'explore_object_rewards_many',
]

psychlab = [
    'multiple_object_tracking',       
    'glass_pattern_detection',  
    'motion_discrimination_easy',     
    'odd_one_out',               
    'temporal_bisection',
    'cued_temporal_production',      
    'harlow',                   
    'motion_discrimination',                           
    'temporal_discrimination',                           
    'landoltC_identification',  
    'multiple_object_tracking_easy',  
    'ready_set_go',              
]

psychlab_memory = [
    'arbitrary_visuomotor_mapping_holdout_extrapolate',  'continuous_recognition_holdout_interpolate',  'explore_goal_locations_interpolate',
    'arbitrary_visuomotor_mapping_holdout_interpolate',  'continuous_recognition_train',                'explore_goal_locations_train_large',
    'arbitrary_visuomotor_mapping_train',                'explore_goal_locations_extrapolate',          'explore_goal_locations_train_small',
    'change_detection_holdout_extrapolate',              'explore_goal_locations_holdout_extrapolate',  'what_then_where_holdout_interpolate',
    'change_detection_holdout_interpolate',              'explore_goal_locations_holdout_interpolate',  'what_then_where_train',
    'change_detection_train',                            'explore_goal_locations_holdout_large',
    'continuous_recognition_holdout_extrapolate',        'explore_goal_locations_holdout_small',
]

psychlab_visuospatial = [
    'memory_guided_saccade',  'odd_one_out',  'pathfinder',  'pursuit',  'visually_guided_antisaccade',  'visually_guided_prosaccade',  'visual_match',
]

fast_mapping = [
    'fast_mapping_train',    'slow_mapping_train',
]

others = [
    'lt_chasm',
    'lt_hallway_slope',
    'lt_horseshoe_color',
    'lt_space_bounce_hard',
    'nav_maze_random_goal_01',
    'nav_maze_random_goal_02',
    'nav_maze_random_goal_03',
    'nav_maze_static_01',
    'nav_maze_static_02',
    'nav_maze_static_03',
    'seekavoid_arena_01',
    'stairway_to_melon',
]



tasksets = {
    'dmlab30': DMLAB_30,
    'others': others,
    'psychlab': psychlab,
    'psychlab_memory': psychlab_memory,
    'psychlab_visuospatial': psychlab_visuospatial,
    'fast_mapping': fast_mapping,
}

tasksets_path = {
    'dmlab30': 'contributed/dmlab30/',
    'others': '',
    'psychlab': 'contributed/psychlab/',
    'psychlab_memory': 'contributed/psychlab/memory_suite_01/',
    'psychlab_visuospatial': 'contributed/psychlab/visuospatial_suite/',
    'fast_mapping': 'contributed/fast_mapping/',
}


# def human_normalized_score(game, returns):
#   """Computes human normalized score.

#   Args:
#     game: The DeepMind Lab game.
#     returns: A list of episode returns.

#   Returns:
#     A float with the human normalized score in percentage.
#   """
#   human = HUMAN_SCORES[game]
#   random = RANDOM_SCORES[game]
#   return (np.mean(returns) - random) / (human - random) * 100
