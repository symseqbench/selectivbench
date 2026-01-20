# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import numpy as np
import torch


class BinaryDataset():
    """ 
    """

    def __init__(
        self,
        vocab_size
    ):  
    
        self.vocab_size = vocab_size
        self.labels = np.arange(self.vocab_size)
        one_hot = np.eye(vocab_size, dtype=int)
        self.stimulus = {k: v for i, (k, v) in enumerate(zip(self.labels, one_hot))}
        x = np.arange(self.vocab_size) 
        self.class_dict = {key: key for key in x}

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, index):

        x = self.stimulus[index]
        x = x[None, :]

        return torch.from_numpy(x), index

