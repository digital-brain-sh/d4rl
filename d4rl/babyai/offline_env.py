from abc import ABC

import gym
import h5py
from tqdm import tqdm
import numpy as np

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.

    Args:
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
        deprecated: If True, will display a warning that the environment is deprecated.
    """

    def __init__(self, game=None, ref_max_score=None, ref_min_score=None, **kwargs):
        super(OfflineEnv, self).__init__(**kwargs)
        self.game = game
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    @property
    def dataset_filepath(self):
        return f'/nfs/dgx08/raid/babyai/{self.game}_agent.hdf5'

    def get_dataset(self, h5path=None):
        if h5path is None:
            h5path = self.dataset_filepath
        data_dict = {}
        obs_names = []
        with h5py.File(h5path, 'r') as dataset_file:
            for k in get_keys(dataset_file):
                if "observations" in k and len(k.split('/')) > 1:
                    obs_name = '/'.join(k.split('/')[1:])
                    if "observations" not in data_dict:
                        data_dict["observations"] = dict()
                    data_dict["observations"][obs_name] = dataset_file[k][:]
                    obs_names.append(obs_name)
                else:
                    try:  # first try loading as an array
                        data_dict[k] = dataset_file[k][:]
                    except ValueError as e:  # try loading as a scalar
                        data_dict[k] = dataset_file[k][()]

        # XXX(ziyu): if need to use a unified API, this line should be changed somewhere else
        data_dict['observations']['mission'] = data_dict['observations']['mission'].astype(str)

        # ziyu: this is because baby AI's h5 file is of wrong dtype
        if "int" not in data_dict['actions'].dtype.name:
            data_dict['actions'] = data_dict['actions'].astype(np.int32)
        # Run a few quick sanity checks
        N_samples = data_dict['rewards'].shape[0]
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        
        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path):
        """
        Returns a slice of the full dataset.

        Args:
            chunk_id (int): An integer representing which slice of the dataset to return.

        Returns:
            A dictionary containing observtions, actions, rewards, and terminals.
        """

        dataset_file = h5py.File(h5path, 'r')

        if 'virtual' not in dataset_file.keys():
            raise ValueError('Dataset is not a chunked dataset')
        available_chunks = [int(_chunk) for _chunk in list(dataset_file['virtual'].keys())]
        if chunk_id not in available_chunks:
            raise ValueError('Chunk id not found: %d. Available chunks: %s' % (chunk_id, str(available_chunks)))

        load_keys = ['observations', 'actions', 'rewards', 'terminals']
        data_dict = {k: dataset_file['virtual/%d/%s' % (chunk_id, k)][:] for k in load_keys}
        dataset_file.close()
        return data_dict


