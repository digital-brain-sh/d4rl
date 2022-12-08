import d4rl
import gym

name_format = "tsp200-{}-expert-v1"

for i in range(1, 32):
    name = name_format.format(i)
    env = gym.make(name)

    mean_total = 0
    for traj in range(50):
        obs = env.reset(traj_idx=traj)

        total_rew = 0
        done = False
        while not done:
            obs, rew, done, info = env.step(0)
            total_rew += rew
        print(total_rew, env.get_normalized_score(total_rew))
    # mean_total += total_rew / 50

    # print("reward:", mean_total, env.get_normalized_score(mean_total))
