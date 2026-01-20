#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the dataloader is defined for the SHD and SSC datasets.
"""
import os
import logging
from pathlib import Path
from os.path import join
from collections import defaultdict

import h5py
import torch
try:
    from torchaudio.transforms import MFCC
    import torchaudio
except ImportError:
    print("Warning: MFCC not loaded")
    pass

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from symseqbench.seqbench.dataset.base import BaseDataset

logger = logging.getLogger(__name__)

SC_labels = [
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go',
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'bed',
    'bird',
    'cat',
    'dog',
    'happy',
    'house',
    'marvin',
    'sheila',
    'tree',
    'wow',
    'backward',
    'forward',
    'follow',
    'learn',
    'visual'
]


class SpeechCommands(BaseDataset):
    """
    Dataset class for the original non-spiking Speech Commands (SC)
    dataset. Generated mel-spectrograms use 40 bins by default.

    Arguments
    ---------
    data_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of the HD dataset, must be either "train" or "test".
    use_augm : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    """

    def __init__(self,
        data_folder,
        split
    ):
        if split not in ["training", "validation", "testing"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder
        EXCEPT_FOLDER = "_background_noise_"

        def load_list(filename):
            filepath = join(self.data_folder, filename)
            with open(filepath) as f:
                return [join(self.data_folder, i.strip()) for i in f]

        self.labels = SC_labels

        if split == "training":
            files = sorted(str(p) for p in Path(data_folder).glob("*/*.wav"))
            exclude = load_list("validation_list.txt") + load_list("testing_list.txt")
            exclude = set(exclude)
            self.file_list = []
            class_dict = defaultdict(list)
            i=0
            for w in files:
                if w not in exclude and EXCEPT_FOLDER not in w:
                    self.file_list.append(w)
                    relpath = os.path.relpath(w, self.data_folder)
                    label, _ = os.path.split(relpath)  
                    class_index = self.labels.index(label)
                    class_dict[class_index].append(i)
                    i += 1            
        else:
            self.file_list = load_list(str(split) + "_list.txt")
            class_dict = defaultdict(list)
            for i, w in enumerate(self.file_list):
                relpath = os.path.relpath(w, self.data_folder)
                label, _ = os.path.split(relpath)  
                class_index = self.labels.index(label)
                class_dict[class_index].append(i)

        self.class_dict = dict(class_dict)

        sample_rate = 16000
        n_mfcc = 13
        n_mels = 40
        n_fft = 512
        win_length = int(sample_rate * 0.025)  # 25ms
        hop_length = int(sample_rate * 0.01)   # 10ms

        self.mfcc = MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "win_length": win_length,
            'window_fn': torch.hann_window
        },
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Read waveform
        filename = self.file_list[index]
        x, _ = torchaudio.load(filename)

        # Compute acoustic features
        x = self.mfcc(x).squeeze().permute(1,0)

        # Get label
        relpath = os.path.relpath(filename, self.data_folder)
        label, _ = os.path.split(relpath)
        y = torch.tensor(self.labels.index(label))

        return x, y

    def generateBatch(self, batch):
        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys

class SpikingDataset(BaseDataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the dataset (h5py file).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        dataset_name,
        data_folder,
        split,
        nb_steps=140,
        max_time=1.4,
        num_bins=1
    ):
        self.nb_steps = nb_steps
        self.num_bins = num_bins
        self.nb_units = 700
        self.nb_units_binned = self.nb_units//self.num_bins
        self.max_time = max_time
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=int)
        
        super().__init__(path=data_folder,
                         inp_enc=dataset_name,
                         split=split)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]
        length = max(times)
        
        x_idx = torch.LongTensor(np.array([times, units]))
        x_val = torch.FloatTensor(np.ones(len(times)))
        x_size = torch.Size([length, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size)
        y = self.labels[index]

        x = x.to_dense()
        T = x.shape[0]
        J = self.nb_units
        Bin = self.num_bins

        # Binning
        with torch.no_grad():
            x = x.contiguous().view(T, J//Bin, Bin).sum(-1)

        return x, y

#    def generateBatch(self, batch):
#        xs, ys = zip(*batch)
#        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
#        xlens = torch.tensor([x.shape[0] for x in xs])
#        ys = torch.LongTensor(ys).to(self.device)
#        return xs, xlens, ys


#def load_shd_or_ssc(
#    dataset_name,
#    data_folder,
#    split,
#    batch_size,
#    nb_steps=100,
#    shuffle=True,
#    workers=0,
#):
#    """
#    This function creates a dataloader for a given split of
#    the SHD or SSC datasets.
#
#    Arguments
#    ---------
#    dataset_name : str
#        Name of the dataset, either shd or ssc.
#    data_folder : str
#        Path to folder containing the Heidelberg Digits dataset.
#    split : str
#        Split of dataset, must be either "train" or "test" for SHD.
#        For SSC, can be "train", "valid" or "test".
#    batch_size : int
#        Number of examples in a single generated batch.
#    shuffle : bool
#        Whether to shuffle examples or not.
#    workers : int
#        Number of workers.
#    """
#    if dataset_name not in ["shd", "ssc"]:
#        raise ValueError(f"Invalid dataset name {dataset_name}")
#
#    if split not in ["train", "valid", "test"]:
#        raise ValueError(f"Invalid split name {split}")
#
#    if dataset_name == "shd" and split == "valid":
#        logging.info("SHD does not have a validation split. Using test split.")
#        split = "test"
#
#    dataset = SpikingDataset(dataset_name, data_folder, split, nb_steps)
#    logging.info(f"Number of examples in {split} set: {len(dataset)}")
#
#    loader = DataLoader(
#        dataset,
#        batch_size=batch_size,
#        collate_fn=dataset.generateBatch,
#        shuffle=shuffle,
#        num_workers=workers,
#        pin_memory=True,
#    )
#
#    return loader
