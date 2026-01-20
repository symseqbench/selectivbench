# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import os
from symseqbench.seqbench.dataset.shd_ssc import SpikingDataset, SpeechCommands
from symseqbench.seqbench.dataset.synthetic import BinaryDataset


def create_base_dataset_from_config(config, name):
    if config['inp_enc'] == 'shd':
        if name == 'valid':
            name = 'test'
        return SpikingDataset(
            'shd',
            config['base_dataset_path'],
            name,
            nb_steps=config['nb_steps'],
            max_time=config['max_time'],
            num_bins=config['num_bins', 1]
        )
    if config['inp_enc'] == 'ssc':
        return SpikingDataset(
            'ssc',
            config['base_dataset_path'],
            name,
            nb_steps=config['nb_steps'],
            max_time=config['max_time'],
            num_bins=config['num_bins', 1]
        )
    elif config['inp_enc'] == 'gsc':
        if name == 'train':
            name = 'training'
        else:
            name = 'testing'
        return SpeechCommands(
            os.path.join(config['base_dataset_path'], 'GSC'),
            name
        )

    elif config['inp_enc'] == "one_hot":
        base_dataset = BinaryDataset(config['vocab_size'])

        return base_dataset

    else:
        raise ValueError(f'Unknown input encoding {inp_enc}!')
