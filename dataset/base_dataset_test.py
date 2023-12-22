from __future__ import division

import numpy as np
import pickle

from torch.utils.data import Dataset

import config
import constants

class BaseDataset_test(Dataset):
    def __init__(self, dataset, dataset_paths, hist_length):
        super(BaseDataset_test, self).__init__()
        
        dataset_path = dataset_paths[dataset]
        with open(dataset_path, 'rb') as f:
            label = pickle.load(f)

        self.label = label
        self.len = len(self.label)
        self.hist_length = hist_length

    def __getitem__(self, index):

        trunk_path = self.label[index]
        anno = np.load(trunk_path+'.npy')
            
        Y = {}
        Y['q']  = anno[(config.hist_length-self.hist_length):, :63]
        Y['shape']  = anno[(config.hist_length-self.hist_length):, 63:63+10]
        Y['gender_id']  = anno[(config.hist_length-self.hist_length):, 63+10]

        return Y

    def __len__(self):
        return self.len