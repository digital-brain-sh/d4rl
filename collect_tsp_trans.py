"""
Collect expert experiences for TSP tasks. Before collection, please ensure you have
picled node embedding file and original dataset file.
"""

import argparse
import gym
import tqdm
import numpy as np

from d4rl.tsp.env import TSPEnv
from d4rl.utils.dataset_utils import TrajectoryDatasetWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect TSP experiences")

    parser.add_argument("--scale", type=int, help="problem scale", default=100, choices={100, 200})
    parser.add_argument("--node-embedding-path", type=str, required=True, help="file path for node embedding.")
    parser.add_argument("--env-dataset", type=str, help="file path for environment dataset (original dataset)", required=True)
    # should be hdf5, e.g. "/home/xxx/expert_traj.hdf5"
    parser.add_argument("--save-traj-path", type=str, help="file path for collected expert trajectories saving", required=True)

    args = parser.parse_args()

    env = TSPEnv(scale=args.scale, action_dim=args.scale, dataset_path=args.env_dataset, node_embedding_path=args.node_embedding_path, use_raw_state=False)
    # load environment dataset
    env.reset()

    num_traj = env.dataset.coords.shape[0]
    high_episode_return = 0
    low_episode_return = 0
    knn_act_dim = 0

    writer = TrajectoryDatasetWriter(n_agent=1)

    for traj_idx in tqdm.tqdm(range(num_traj), desc="Data Trans"):
        obs = env.reset(traj_idx=traj_idx)
        done = False
        cnt = 1

        while not done:
            expert_choice = env.dataset.soln[env.sequence_idx][cnt]
            # nodex idx is the expected action
            action = env.get_cur_ava_nodes().tolist().index(expert_choice)
            next_obs, rew, done, info = env.step(action)
            writer.append_data(obs, action, rew, done)
            obs = next_obs
    
            high_episode_return += rew
            knn_act_dim = max(action, knn_act_dim)

        done = False
        # use random generated traj as the lowest score
        while not done:
            ava_mask = env.get_cur_action_mask()
            candidates = np.nonzero(ava_mask)[0]
            action = np.random.choice(candidates)
            _, rew, done, _ = env.step(action)
            low_episode_return += rew

    writer.write_dataset(args.save_traj_path)
    print(f"knn_action_dim: {knn_act_dim + 1}, high return: {high_episode_return}, low return: {low_episode_return}")