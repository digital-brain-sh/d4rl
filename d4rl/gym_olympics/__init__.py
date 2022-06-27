from gym.envs.registration import register


# olympics running env with the same map, map id can be specified as
# gym.make('gym_olympics:running-v0', map_id=1)
register(
    id='running-v0',
    entry_point='gym_olympics.olympics_offline_envs:get_running',
    max_episode_steps=400,
    # kwargs={
    #     'maze_spec': '',
    #     'reward_type': '',
    #     'reset_target': '',
    #     'ref_min_score': '',
    #     'ref_max_score': '',
    #     'dataset_url': '',
    # }
)

# olympics running env with map changing randomly
# gym.make('gym_olympics:rd_running-v0')
register(
    id='rd_running-v0',
    entry_point='gym_olympics.olympics_offline_envs:get_rd_running',
    max_episode_steps=400,
    # kwargs={
    #     'maze_spec': '',
    #     'reward_type': '',
    #     'reset_target': '',
    #     'ref_min_score': '',
    #     'ref_max_score': '',
    #     'dataset_url': '',
    # }
)

# gym.make('gym_olympics:table_hockey-v0')
register(
    id='table_hockey-v0',
    entry_point='gym_olympics.olympics_offline_envs:get_table_hockey',
    max_episode_steps=400,
    # kwargs={
    #     'maze_spec': '',
    #     'reward_type': '',
    #     'reset_target': '',
    #     'ref_min_score': '',
    #     'ref_max_score': '',
    #     'dataset_url': '',
    # }
)

# gym.make('gym_olympics:football-v0')
register(
    id='football-v0',
    entry_point='gym_olympics.olympics_offline_envs:get_football',
    max_episode_steps=400,
    # kwargs={
    #     'maze_spec': '',
    #     'reward_type': '',
    #     'reset_target': '',
    #     'ref_min_score': '',
    #     'ref_max_score': '',
    #     'dataset_url': '',
    # }
)

# gym.make('gym_olympics:wrestling-v0')
register(
    id='wrestling-v0',
    entry_point='gym_olympics.olympics_offline_envs:get_wrestling',
    max_episode_steps=400,
    # kwargs={
    #     'maze_spec': '',
    #     'reward_type': '',
    #     'reset_target': '',
    #     'ref_min_score': '',
    #     'ref_max_score': '',
    #     'dataset_url': '',
    # }
)