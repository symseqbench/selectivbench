# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import os
import sys
import time
import random
import logging
import argparse
from os.path import join

import torch
import numpy as np

from symseqbench.seqbench.utils.config import Config
from symseqbench.seqbench.utils import prepare_config
from symseqbench.seqbench.seq_dataset import create_seq_dataset_from_config, PadSequence

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('read_dataset')

def parse_cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the .yaml which contains all user defined parameters.')

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = parse_cli_arguments()
    config = Config.parse_config_from_args(args)
    config.print_config()
    
    config['config_file_path'] = args['config']
    config = prepare_config(config)
    seed = config['seed']
    config['do_classify'] = True

    random.seed(seed)
    np.random.seed(seed)

    config['dataset_size'] = config['dataset_size_train']
    seq_dataset = create_seq_dataset_from_config(config, 'train')

    seq_loader = torch.utils.data.DataLoader(
        seq_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=PadSequence(
            do_classify=config['do_classify'],
            pad_index=-1
        ),
        num_workers=1
    )  

    for entry in seq_loader:
        print('~~~')
        print(entry['data'].shape)
        print(entry['labels'].shape)
        print('min label', torch.min(entry['labels']))
        print('max label', torch.max(entry['labels']))
        print(entry['mask'].shape)
        print(entry['lens'].shape)
        print(entry['gap_mask'].shape)
