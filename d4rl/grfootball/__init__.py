from gym.envs.registration import register


scenario_configs = {
    '11_vs_11_easy_stochastic': {
        'map_name': '11_vs_11_easy_stochastic',
        'n_right_players': 11,
        'n_left_players': 11,
        'play_with_bot': False
    },
    '5_vs_5': {
        'map_name': '5_vs_5',
        'n_right_players': 5,
        'n_left_players': 5,
        'play_with_bot': False
    },
    '5_vs_bot': {
        'map_name': '5_vs_5',
        'n_right_players': 5,
        'n_left_players': 5,
        'play_with_bot': True
    },
    '1_vs_1_easy': {
        'map_name': '1_vs_1_easy',
        'n_right_players': 1,
        'n_left_players': 1,
        'play_with_bot': False
    },
    'academy_counterattack_easy': {
        'map_name': 'academy_counterattack_easy',
        'n_right_players': 1,
        'n_left_players': 1,
        'play_with_bot': False
    }
}


register(
    id='GRFootball-v0',
    entry_point='d4rl.grfootball.grf_envs:get_env'
)