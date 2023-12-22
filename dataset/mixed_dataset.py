import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, hist_length, dataset_name='AMASS', **kwargs):
        # dataset_name = 'H36M', 'AMASS', 'PW3D'

        self.datasets = BaseDataset(dataset_name, hist_length, **kwargs)
        self.length = len(self.datasets)
        print(dataset_name+' dataset size: %d' % len(self.datasets))

    def __getitem__(self, index):
        return self.datasets[index % len(self.datasets)]

    def __len__(self):
        return self.length
