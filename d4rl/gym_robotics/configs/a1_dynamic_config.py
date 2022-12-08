from legged_gym.envs.a1.a1_config import A1RoughCfg


class DynamicRough(A1RoughCfg):

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.