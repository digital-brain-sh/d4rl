from gym.envs.registration import register, registry
from .dataset_info import DATASET_URLS, REF_MIN_SCORE, REF_MAX_SCORE


for task_and_config, uri in DATASET_URLS.items():
    env_id = f"robotics-{task_and_config}-expert-v1"
    task, config = task_and_config.split("-")
    register(
        id=env_id,
        entry_point='d4rl.gym_robotics.env:OfflineRobotics',
        max_episode_steps=1000,
        kwargs={
            'task_name': task,
            'config': config,
            'dataset_url': uri,
            'ref_min_score': REF_MIN_SCORE[task_and_config],
            'ref_max_score': REF_MAX_SCORE[task_and_config]
        }
    )

ALL_ENVS = list(map(lambda x: f"robotics-{x}-expert-v1", DATASET_URLS.keys()))