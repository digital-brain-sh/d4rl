from gym.envs.registration import register


# olympics running env with the same map, map id can be specified as
# gym.make('gym_olympics:running-v0', map_id=1)
register(
    id='running-v0',
    entry_point='gym_olympics.envs:running',
)

# olympics running env with map changing randomly
# gym.make('gym_olympics:rd_running-v0')
register(
    id='rd_running-v0',
    entry_point='gym_olympics.envs:rd_running',
)

# gym.make('gym_olympics:table_hockey-v0')
register(
    id='table_hockey-v0',
    entry_point='gym_olympics.envs:table_hockey',
)

# gym.make('gym_olympics:football-v0')
register(
    id='football-v0',
    entry_point='gym_olympics.envs:football',
)

# gym.make('gym_olympics:wrestling-v0')
register(
    id='wrestling-v0',
    entry_point='gym_olympics.envs:wrestling',
)