# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import os
import pickle
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self,
                 path,
                 inp_enc,
                 split):
        self.path_data = path
        self.inp_enc = inp_enc
        self.split = split
        self.class_dict = self.extract_class_indexes()

    def extract_class_indexes(self):
        sample0 = self[0]
        target0 = sample0[1]

        # Check if the value is already an integer
        if isinstance(target0, int):
            extract_int = lambda x: x
        elif isinstance(target0, np.integer):
            extract_int = lambda x: int(x)
        # Check if the value is a size-1 torch tensor with integer data
        elif isinstance(target0, torch.Tensor) and target0.numel() == 1 and target0.dtype in (torch.int32, torch.int64):
            extract_int = lambda x: x.item()
        # Check if the value is a size-1 numpy array with integer data
        elif isinstance(target0, np.ndarray) and target0.size == 1 and np.issubdtype(target0.dtype, np.integer):
            extract_int = lambda x: int(x.item())
        else:
            raise AttributeError("The dataset must return data that includes (sample, target, ...) and the target to be an integer, a size-1 tensor, or a size-1 array containing an integer. If this is not the case, please provide the class indexes dictionary manually.")

        # Initialize a default dictionary to hold lists of indexes for each class
        class_indexes = defaultdict(list)

        # Loop through all targets and append the index to the corresponding class key
        #for idx, sample in tqdm(enumerate(self), total=len(self), desc="Extracting class indexes"):
        #    class_indexes[extract_int(sample[1])].append(idx)
        

        try:
            with open(os.path.join(self.path_data, self.inp_enc + '_' + self.split) + '.pkl', "rb") as f:
                class_indexes = pickle.load(f)
                print('class_indexes loaded for:', self.inp_enc , "from:", self.path_data)
                #class_indexes = np.load(os.path.join(self.params['path_data'],
                #                                     self.params['inp_enc']) + '.npy', 
                #                        allow_pickle=True)
            return class_indexes
        except:    
            print('class_indexes doesnt exist for:', self.inp_enc + '_' + self.split)
            for idx, sample in tqdm(enumerate(self),
                                    total=len(self),
                                    desc="Extracting class indexes"):
                class_indexes[extract_int(sample[1])].append(idx)

            # Save it to a file
            os.makedirs(self.path_data, exist_ok=True)
            print('Saving class_indexes for:', self.inp_enc + '_' + self.split, "in:", self.path_data)
            with open(os.path.join(self.path_data, self.inp_enc + '_' + self.split) + '.pkl', "wb") as f:
                pickle.dump(class_indexes, f)
  
            return dict(class_indexes)
