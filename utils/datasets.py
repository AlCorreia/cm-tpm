from torch.utils.data import Dataset
import numpy as np
import subprocess
import csv
import os


DEBD_DATASETS = [
    'nltcs',
    'msnbc',
    'kdd',
    'plants',
    'baudio',
    'jester',
    'bnetflix',
    'accidents',
    'tretail',
    'pumsb_star',
    'dna',
    'kosarek',
    'msweb',
    'book',
    'tmovie',
    'cwebkb',
    'cr52',
    'c20ng',
    'bbc',
    'ad',
]


def maybe_download_debd(root: str):
    if not os.path.isdir(root):
        subprocess.run(['git', 'clone', 'https://github.com/UCLA-StarAI/Density-Estimation-Datasets', root])
        for f in os.listdir(root):
            if '.' in f and '.git' not in f:
                os.remove(os.path.join(root, f))


def load_debd(
    dataset_name: str,
    dtype: str = 'float32',
    root: str = '../data/debd/'
):
    maybe_download_debd(root)

    train_path = os.path.join(root, 'datasets', dataset_name, dataset_name + '.train.data')
    valid_path = os.path.join(root, 'datasets', dataset_name, dataset_name + '.valid.data')
    test_path = os.path.join(root, 'datasets', dataset_name, dataset_name + '.test.data')

    delimiter = ' ' if dataset_name == 'binarized_mnist' else ','
    train = np.array(list(csv.reader(open(train_path, 'r'), delimiter=delimiter))).astype(dtype)
    valid = np.array(list(csv.reader(open(valid_path, 'r'), delimiter=delimiter))).astype(dtype)
    test = np.array(list(csv.reader(open(test_path, 'r'), delimiter=delimiter))).astype(dtype)

    return train, valid, test


class TensorDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i] if self.transform is None else self.transform(self.data[i])
    
    
class UnsupervisedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        datapoint = self.base[idx]
        if isinstance(datapoint, tuple):
            datapoint, label = datapoint
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint
