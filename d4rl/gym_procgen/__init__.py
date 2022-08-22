from gym.envs.registration import register

REF_MIN_SCORE = {}

REF_MAX_SCORE = {}

DATASET_URLS = {}
ALL_ENVS = []

game_settings = {
    "bigfish": ["procgen:procgen-bigfish-v0", "hard", 20.],
    "bossfight": ["procgen:procgen-bossfight-v0", "hard", 10.],
    "caveflyer": ["procgen:procgen-caveflyer-v0", "hard", 7.5],
    "chaser": ["procgen:procgen-chaser-v0", "hard", 7.],
    "climber": ["procgen:procgen-climber-v0", "hard", 8.],
    "coinrun": ["procgen:procgen-coinrun-v0", "hard", 8.],
    "dodgeball": ["procgen:procgen-dodgeball-v0", "hard", 8.],
    "fruitbot": ["procgen:procgen-fruitbot-v0", "hard", 18.],
    "heist": ["procgen:procgen-heist-v0", "easy", 7.5],
    "jumper": ["procgen:procgen-jumper-v0", "hard", 5.],
    "leaper": ["procgen:procgen-leaper-v0", "hard", 7.5],
    "maze": ["procgen:procgen-maze-v0", "easy", 7.5],
    "miner": ["procgen:procgen-miner-v0", "hard", 15.],
    "ninja": ["procgen:procgen-ninja-v0", "hard", 8.],
    "plunder": ["procgen:procgen-plunder-v0", "hard", 15.],
    "starpilot": ["procgen:procgen-starpilot-v0", "hard", 15.],
}

for game in ['bigfish',
             'bossfight',
             'caveflyer',
             'chaser',
             'climber',
             'coinrun',
             'dodgeball',
             'fruitbot',
             'heist',
             'jumper',
             'leaper'
             'maze',
             'miner',
             'ninja',
             'plunder',
             'starpilot',
             ]:
    env_name = '%s-expert-v0' % game
    ALL_ENVS.append(env_name)
    register(
        id=env_name,
        entry_point='d4rl.gym_procgen.envs:OfflineProcgenEnv',
        kwargs={
            'game': game_settings[game][0],
            'ref_min_score': None,
            'ref_max_score': None,
            'start_level': 0,
            'num_levels': 100000,
            'distribution_mode': game_settings[game][1],
        }
    )
