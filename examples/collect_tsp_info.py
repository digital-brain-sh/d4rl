"""
Collect tsp environment info, including min_score_ref, max_score_ref and action dim
"""

import os
import argparse
import numpy as np
import tqdm
import json

from d4rl.tsp.env import TSPEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    args = parser.parse_args()

    action_dim_info = dict()
    high_score_info = dict()
    low_score_info = dict()

    for fname in os.listdir(args.dataset_dir):
        if not fname.endswith("pkl"):
            continue
        dataset_path = os.path.join(args.dataset_dir, fname)
        task_seed = int(fname[:-4].split("_")[2][4:])
        env = TSPEnv(scale=args.scale, action_dim=args.scale, dataset_path=dataset_path, node_embedding_path="/raid/gato_dataset/tsp200/tsp200_77_node_embeddings.npy")
        high_rew = -np.inf
        low_rew = np.inf
        action_dim = 0
        # load dataset before running
        env.reset()

        for i in range(100):
            obs = env.reset(traj_idx=i)
            solutions = env.dataset.soln[i]
            done = False
            cnt = 0
            traj_rew = 0
            while not done:
                cnt += 1
                node = solutions[cnt]
                node_idx = env.get_cur_ava_nodes().tolist().index(node)
                # if node_idx > 60:
                #     import pdb; pdb.set_trace()
                #     print(node_idx)
                action_dim = max(action_dim, node_idx + 1)
                next_obs, rew, done, info = env.step(node_idx)
                traj_rew += rew
                obs = next_obs
            high_rew = max(traj_rew, high_rew)

        for i in range(100):
            obs = env.reset(traj_idx=i)
            solutions = env.dataset.soln[i]
            done = False
            cnt = 0
            traj_rew = 0
            while not done:
                cnt += 1
                candidates = np.nonzero(env.get_cur_action_mask())[0].tolist()
                node_idx = np.random.choice(candidates)
                # if node_idx > 60:
                #     import pdb; pdb.set_trace()
                #     print(node_idx)
                next_obs, rew, done, info = env.step(node_idx)
                traj_rew += rew
                obs = next_obs
            low_rew = min(traj_rew, low_rew)

        high_score_info[f"TSP{args.scale}/{task_seed}"] = high_rew
        low_score_info[f"TSP{args.scale}/{task_seed}"] = low_rew
        action_dim_info[task_seed] = action_dim

        print("high_score:", high_score_info)
        print("low_score:", low_score_info)
        print("action_dim:", action_dim_info)

    with open(f"/home/mingzhou/d4rl/d4rl/tsp/tsp{args.scale}-city-action-dim.json", 'w') as f:
        json.dump(action_dim_info, f)

    with open(f"/home/mingzhou/d4rl/d4rl/tsp/tsp{args.scale}-city-expert-info.json", 'w') as f:
        json.dump({
            "REF_MIN_SCORES": low_score_info,
            "REF_MAX_SCORES": high_score_info
        }, f)


