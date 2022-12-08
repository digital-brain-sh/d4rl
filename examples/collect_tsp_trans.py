"""
Collect expert experiences for TSP tasks. Before collection, please ensure you have
picled node embedding file and original dataset file.
"""

import argparse
import os
import gym
import tqdm
import numpy as np
import ray

from ray.util import ActorPool
from d4rl.tsp.env import TSPEnv
from d4rl.utils.dataset_utils import TrajectoryDatasetWriter


@ray.remote(num_cpus=0.1)
def env_runner(env, traj_idx_start, traj_idx_end):
    obs_list, action_list, rew_list, done_list = [], [], [], []
    for traj_idx in range(traj_idx_start, traj_idx_end):
        obs = env.reset(traj_idx=traj_idx)
        done = False
        cnt = 1

        while not done:
            expert_choice = env.dataset.soln[env.sequence_idx][cnt]
            # nodex idx is the expected action
            action = env.get_cur_ava_nodes().tolist().index(expert_choice)
            next_obs, rew, done, info = env.step(action)
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            action_list.append(action)
            # writer.append_data(obs, action, rew, done)
            obs = next_obs
            cnt += 1

    return {
        "observations": obs_list,
        "actions": action_list,
        "rewards": rew_list,
        "terminals": done_list
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect TSP experiences")

    parser.add_argument("--scale", type=int, help="problem scale", required=True, choices={100, 200})
    parser.add_argument("--env-dataset-dir", type=str, help="original dataset, including node-embedding path and environment dataset path.", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--rl-dataset-dir", type=str, help="destinated directory for RL trajectories storage.")
    args = parser.parse_args()

    env_dataset_dir = args.env_dataset_dir
    actor_num = 10

    if not ray.is_initialized():
        ray.init()
    actor_pool = ActorPool([env_runner.remote for _ in range(actor_num)])
    fnames = [f"tsp200_city_seed{i}.pkl" for i in range(args.start, args.end)]
    for fname in fnames: # os.listdir(env_dataset_dir):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(env_dataset_dir, fname)
        names = fname.split("_")
        names = names[:2] + [names[2][4:-4], "node_embeddings.npy"]
        node_embeding_fname = "_".join(names)
        node_embedding_path = os.path.join(env_dataset_dir, node_embeding_fname)

        print(f"[*] created environment for: {fpath}\n\twith node_embedding: {node_embedding_path}")
        env = TSPEnv(scale=args.scale, action_dim=args.scale, dataset_path=fpath, node_embedding_path=node_embedding_path, use_raw_state=False)
        # load environment dataset
        env.reset()

        num_traj = env.dataset.coords.shape[0]
        assert num_traj % actor_num == 0, (num_traj, actor_num)

        writer = TrajectoryDatasetWriter(n_agent=1)
        for traj_idx in tqdm.tqdm(range(0, num_traj, actor_num * actor_num), desc="Data Trans"):
            res = actor_pool.map(lambda a, v: a(env, v, v + actor_num), list(range(traj_idx, traj_idx + actor_num * actor_num, actor_num)))
            for data_dict in res:
                for k, v in data_dict.items():
                    writer.data[k].extend(v)

        path_seg = fpath.split("/")
        path_seg[-2] += "_expert"
        path_seg[-1] = fname.split(".")[0] + ".hdf5"
        save_traj_path = os.path.join("/", *path_seg)
        print("[*] write dataset to:", save_traj_path)
        writer.write_dataset(save_traj_path)
