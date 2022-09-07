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

DMLAB_30_psych = [
    'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition',
    'psychlab_sequential_comparison',
    'psychlab_visual_search',
]

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

gen_explore_goal_locations = [
'explore_goal_locations_0', 'explore_goal_locations_1', 'explore_goal_locations_10', 'explore_goal_locations_11', 'explore_goal_locations_12', 'explore_goal_locations_13', 'explore_goal_locations_14', 'explore_goal_locations_15', 'explore_goal_locations_16', 'explore_goal_locations_17', 'explore_goal_locations_18', 'explore_goal_locations_19', 'explore_goal_locations_2', 'explore_goal_locations_20', 'explore_goal_locations_21', 'explore_goal_locations_22', 'explore_goal_locations_23', 'explore_goal_locations_24', 'explore_goal_locations_25', 'explore_goal_locations_26', 'explore_goal_locations_27', 'explore_goal_locations_28', 'explore_goal_locations_29', 'explore_goal_locations_3', 'explore_goal_locations_30', 'explore_goal_locations_31', 'explore_goal_locations_32', 'explore_goal_locations_33', 'explore_goal_locations_34', 'explore_goal_locations_35', 'explore_goal_locations_36', 'explore_goal_locations_37', 'explore_goal_locations_38', 'explore_goal_locations_39', 'explore_goal_locations_4', 'explore_goal_locations_40', 'explore_goal_locations_41', 'explore_goal_locations_42', 'explore_goal_locations_43', 'explore_goal_locations_44', 'explore_goal_locations_45', 'explore_goal_locations_46', 'explore_goal_locations_47', 'explore_goal_locations_48', 'explore_goal_locations_49', 'explore_goal_locations_5', 'explore_goal_locations_50', 'explore_goal_locations_51', 'explore_goal_locations_52', 'explore_goal_locations_53', 'explore_goal_locations_54', 'explore_goal_locations_55', 'explore_goal_locations_56', 'explore_goal_locations_57', 'explore_goal_locations_58', 'explore_goal_locations_59', 'explore_goal_locations_6', 'explore_goal_locations_60', 'explore_goal_locations_61', 'explore_goal_locations_62', 'explore_goal_locations_63', 'explore_goal_locations_64', 'explore_goal_locations_65', 'explore_goal_locations_66', 'explore_goal_locations_67', 'explore_goal_locations_68', 'explore_goal_locations_69', 'explore_goal_locations_7', 'explore_goal_locations_70', 'explore_goal_locations_71', 'explore_goal_locations_8', 'explore_goal_locations_9'
]

gen_explore_object_locations = [
'explore_object_locations_0', 'explore_object_locations_1', 'explore_object_locations_10', 'explore_object_locations_11', 'explore_object_locations_12', 'explore_object_locations_13', 'explore_object_locations_14', 'explore_object_locations_15', 'explore_object_locations_16', 'explore_object_locations_17', 'explore_object_locations_18', 'explore_object_locations_19', 'explore_object_locations_2', 'explore_object_locations_20', 'explore_object_locations_21', 'explore_object_locations_22', 'explore_object_locations_23', 'explore_object_locations_24', 'explore_object_locations_25', 'explore_object_locations_26', 'explore_object_locations_27', 'explore_object_locations_28', 'explore_object_locations_29', 'explore_object_locations_3', 'explore_object_locations_30', 'explore_object_locations_31', 'explore_object_locations_32', 'explore_object_locations_33', 'explore_object_locations_34', 'explore_object_locations_35', 'explore_object_locations_36', 'explore_object_locations_37', 'explore_object_locations_38', 'explore_object_locations_39', 'explore_object_locations_4', 'explore_object_locations_40', 'explore_object_locations_41', 'explore_object_locations_42', 'explore_object_locations_43', 'explore_object_locations_44', 'explore_object_locations_45', 'explore_object_locations_46', 'explore_object_locations_47', 'explore_object_locations_5', 'explore_object_locations_6', 'explore_object_locations_7', 'explore_object_locations_8', 'explore_object_locations_9'
]

gen_explore_object_rewards = [
'explore_object_rewards_0', 'explore_object_rewards_1', 'explore_object_rewards_10', 'explore_object_rewards_11', 'explore_object_rewards_12', 'explore_object_rewards_13', 'explore_object_rewards_14', 'explore_object_rewards_15', 'explore_object_rewards_16', 'explore_object_rewards_17', 'explore_object_rewards_18', 'explore_object_rewards_19', 'explore_object_rewards_2', 'explore_object_rewards_20', 'explore_object_rewards_21', 'explore_object_rewards_22', 'explore_object_rewards_23', 'explore_object_rewards_24', 'explore_object_rewards_25', 'explore_object_rewards_26', 'explore_object_rewards_27', 'explore_object_rewards_28', 'explore_object_rewards_29', 'explore_object_rewards_3', 'explore_object_rewards_30', 'explore_object_rewards_31', 'explore_object_rewards_32', 'explore_object_rewards_33', 'explore_object_rewards_34', 'explore_object_rewards_35', 'explore_object_rewards_36', 'explore_object_rewards_37', 'explore_object_rewards_38', 'explore_object_rewards_39', 'explore_object_rewards_4', 'explore_object_rewards_40', 'explore_object_rewards_41', 'explore_object_rewards_42', 'explore_object_rewards_43', 'explore_object_rewards_44', 'explore_object_rewards_45', 'explore_object_rewards_46', 'explore_object_rewards_47', 'explore_object_rewards_48', 'explore_object_rewards_49', 'explore_object_rewards_5', 'explore_object_rewards_50', 'explore_object_rewards_51', 'explore_object_rewards_52', 'explore_object_rewards_53', 'explore_object_rewards_54', 'explore_object_rewards_55', 'explore_object_rewards_56', 'explore_object_rewards_57', 'explore_object_rewards_58', 'explore_object_rewards_59', 'explore_object_rewards_6', 'explore_object_rewards_60', 'explore_object_rewards_61', 'explore_object_rewards_62', 'explore_object_rewards_63', 'explore_object_rewards_64', 'explore_object_rewards_65', 'explore_object_rewards_66', 'explore_object_rewards_67', 'explore_object_rewards_68', 'explore_object_rewards_69', 'explore_object_rewards_7', 'explore_object_rewards_70', 'explore_object_rewards_71', 'explore_object_rewards_72', 'explore_object_rewards_73', 'explore_object_rewards_74', 'explore_object_rewards_75', 'explore_object_rewards_76', 'explore_object_rewards_77', 'explore_object_rewards_78', 'explore_object_rewards_79', 'explore_object_rewards_8', 'explore_object_rewards_80', 'explore_object_rewards_9'
]

gen_explore_obstructed_goals = [
'explore_obstructed_goals_0', 'explore_obstructed_goals_1', 'explore_obstructed_goals_10', 'explore_obstructed_goals_100', 'explore_obstructed_goals_101', 'explore_obstructed_goals_102', 'explore_obstructed_goals_103', 'explore_obstructed_goals_104', 'explore_obstructed_goals_105', 'explore_obstructed_goals_106', 'explore_obstructed_goals_107', 'explore_obstructed_goals_11', 'explore_obstructed_goals_12', 'explore_obstructed_goals_13', 'explore_obstructed_goals_14', 'explore_obstructed_goals_15', 'explore_obstructed_goals_16', 'explore_obstructed_goals_17', 'explore_obstructed_goals_18', 'explore_obstructed_goals_19', 'explore_obstructed_goals_2', 'explore_obstructed_goals_20', 'explore_obstructed_goals_21', 'explore_obstructed_goals_22', 'explore_obstructed_goals_23', 'explore_obstructed_goals_24', 'explore_obstructed_goals_25', 'explore_obstructed_goals_26', 'explore_obstructed_goals_27', 'explore_obstructed_goals_28', 'explore_obstructed_goals_29', 'explore_obstructed_goals_3', 'explore_obstructed_goals_30', 'explore_obstructed_goals_31', 'explore_obstructed_goals_32', 'explore_obstructed_goals_33', 'explore_obstructed_goals_34', 'explore_obstructed_goals_35', 'explore_obstructed_goals_36', 'explore_obstructed_goals_37', 'explore_obstructed_goals_38', 'explore_obstructed_goals_39', 'explore_obstructed_goals_4', 'explore_obstructed_goals_40', 'explore_obstructed_goals_41', 'explore_obstructed_goals_42', 'explore_obstructed_goals_43', 'explore_obstructed_goals_44', 'explore_obstructed_goals_45', 'explore_obstructed_goals_46', 'explore_obstructed_goals_47', 'explore_obstructed_goals_48', 'explore_obstructed_goals_49', 'explore_obstructed_goals_5', 'explore_obstructed_goals_50', 'explore_obstructed_goals_51', 'explore_obstructed_goals_52', 'explore_obstructed_goals_53', 'explore_obstructed_goals_54', 'explore_obstructed_goals_55', 'explore_obstructed_goals_56', 'explore_obstructed_goals_57', 'explore_obstructed_goals_58', 'explore_obstructed_goals_59', 'explore_obstructed_goals_6', 'explore_obstructed_goals_60', 'explore_obstructed_goals_61', 'explore_obstructed_goals_62', 'explore_obstructed_goals_63', 'explore_obstructed_goals_64', 'explore_obstructed_goals_65', 'explore_obstructed_goals_66', 'explore_obstructed_goals_67', 'explore_obstructed_goals_68', 'explore_obstructed_goals_69', 'explore_obstructed_goals_7', 'explore_obstructed_goals_70', 'explore_obstructed_goals_71', 'explore_obstructed_goals_72', 'explore_obstructed_goals_73', 'explore_obstructed_goals_74', 'explore_obstructed_goals_75', 'explore_obstructed_goals_76', 'explore_obstructed_goals_77', 'explore_obstructed_goals_78', 'explore_obstructed_goals_79', 'explore_obstructed_goals_8', 'explore_obstructed_goals_80', 'explore_obstructed_goals_81', 'explore_obstructed_goals_82', 'explore_obstructed_goals_83', 'explore_obstructed_goals_84', 'explore_obstructed_goals_85', 'explore_obstructed_goals_86', 'explore_obstructed_goals_87', 'explore_obstructed_goals_88', 'explore_obstructed_goals_89', 'explore_obstructed_goals_9', 'explore_obstructed_goals_90', 'explore_obstructed_goals_91', 'explore_obstructed_goals_92', 'explore_obstructed_goals_93', 'explore_obstructed_goals_94', 'explore_obstructed_goals_95', 'explore_obstructed_goals_96', 'explore_obstructed_goals_97', 'explore_obstructed_goals_98', 'explore_obstructed_goals_99'
]

gen_lasertag = [
'lasertag_0', 'lasertag_1', 'lasertag_10', 'lasertag_100', 'lasertag_101', 'lasertag_102', 'lasertag_103', 'lasertag_104', 'lasertag_105', 'lasertag_106', 'lasertag_107', 'lasertag_11', 'lasertag_12', 'lasertag_13', 'lasertag_14', 'lasertag_15', 'lasertag_16', 'lasertag_17', 'lasertag_18', 'lasertag_19', 'lasertag_2', 'lasertag_20', 'lasertag_21', 'lasertag_22', 'lasertag_23', 'lasertag_24', 'lasertag_25', 'lasertag_26', 'lasertag_27', 'lasertag_28', 'lasertag_29', 'lasertag_3', 'lasertag_30', 'lasertag_31', 'lasertag_32', 'lasertag_33', 'lasertag_34', 'lasertag_35', 'lasertag_36', 'lasertag_37', 'lasertag_38', 'lasertag_39', 'lasertag_4', 'lasertag_40', 'lasertag_41', 'lasertag_42', 'lasertag_43', 'lasertag_44', 'lasertag_45', 'lasertag_46', 'lasertag_47', 'lasertag_48', 'lasertag_49', 'lasertag_5', 'lasertag_50', 'lasertag_51', 'lasertag_52', 'lasertag_53', 'lasertag_54', 'lasertag_55', 'lasertag_56', 'lasertag_57', 'lasertag_58', 'lasertag_59', 'lasertag_6', 'lasertag_60', 'lasertag_61', 'lasertag_62', 'lasertag_63', 'lasertag_64', 'lasertag_65', 'lasertag_66', 'lasertag_67', 'lasertag_68', 'lasertag_69', 'lasertag_7', 'lasertag_70', 'lasertag_71', 'lasertag_72', 'lasertag_73', 'lasertag_74', 'lasertag_75', 'lasertag_76', 'lasertag_77', 'lasertag_78', 'lasertag_79', 'lasertag_8', 'lasertag_80', 'lasertag_81', 'lasertag_82', 'lasertag_83', 'lasertag_84', 'lasertag_85', 'lasertag_86', 'lasertag_87', 'lasertag_88', 'lasertag_89', 'lasertag_9', 'lasertag_90', 'lasertag_91', 'lasertag_92', 'lasertag_93', 'lasertag_94', 'lasertag_95', 'lasertag_96', 'lasertag_97', 'lasertag_98', 'lasertag_99'
]

tasksets = {
    'dmlab30': DMLAB_30,
    'others': others,
    'psychlab': psychlab,
    'psychlab_memory': psychlab_memory,
    'psychlab_visuospatial': psychlab_visuospatial,
    'fast_mapping': fast_mapping,
    'gen_explore_goal_locations': gen_explore_goal_locations,
    'gen_explore_object_locations': gen_explore_object_locations,
    'gen_explore_object_rewards': gen_explore_object_rewards,
    'gen_explore_obstructed_goals': gen_explore_obstructed_goals,
    'gen_lasertag': gen_lasertag,
}

tasksets_path = {
    'dmlab30': 'contributed/dmlab30/',
    'others': '',
    'psychlab': 'contributed/psychlab/',
    'psychlab_memory': 'contributed/psychlab/memory_suite_01/',
    'psychlab_visuospatial': 'contributed/psychlab/visuospatial_suite/',
    'fast_mapping': 'contributed/fast_mapping/',
    'gen_explore_goal_locations': 'explore_goal_locations/',
    'gen_explore_object_locations': 'explore_object_locations/',
    'gen_explore_object_rewards': 'explore_object_rewards/',
    'gen_explore_obstructed_goals': 'explore_obstructed_goals/',
    'gen_lasertag': 'lasertag/',
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
