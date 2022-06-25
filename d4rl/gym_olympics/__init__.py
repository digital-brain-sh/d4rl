from gym.envs.registration import register

register(
    id='running-v0',
    entry_point='gym_olympics.envs:running',
)

register(
    id='table_hockey-v0',
    entry_point='gym_olympics.envs:table_hockey',
)

register(
    id='football-v0',
    entry_point='gym_olympics.envs:football',
)

register(
    id='wrestling-v0',
    entry_point='gym_olympics.envs:wrestling',
)