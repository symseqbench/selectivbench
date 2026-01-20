# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import os
import sys
import time
import random
import logging
import argparse

import numpy as np

from symseqbench.seqbench.utils.config import Config
from symseqbench.seqbench.seq_dataset import DatasetGenerator
from symseqbench.seqbench import create_base_dataset_from_config
from symseqbench.seqbench.utils import prepare_config, get_config_hash
from symseqbench.seqbench.generator import SequenceGenerator

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('create_dataset')


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

    # Creating training dataset
    config['dataset_size'] = config['dataset_size_train']
    config_hash = get_config_hash(config)
    dataset_root = config['dataset_output_dir']
    dataset_root = f'{dataset_root}-{config_hash}'

    random.seed(seed)
    np.random.seed(seed)

    if 'timestep' in dict(config):
        timestep = config['timestep']
    elif config['inp_enc'] == 'shd':
        timestep = config['max_time']*1000/config['nb_steps']
    else:
        print("timestep is not defined")
        raise

    seq_generator = SequenceGenerator(config)

    dataset_generator = DatasetGenerator(
        seq_generator, timestep,
        dataset_size=config['dataset_size'],
        output_dir=dataset_root,
        config_file_path=config['config_file_path'],
        generate_train=True,
        generate_test=False,
    )

    dataset_generator.generate()

    # Creating testing dataset ...
    config['dataset_size'] = config['dataset_size_test']
    config_hash = get_config_hash(config)
    dataset_root = config['dataset_output_dir']
    dataset_root = f'{dataset_root}-{config_hash}'

    random.seed(seed)
    np.random.seed(seed)

    seq_generator = SequenceGenerator(config)

    dataset_generator = DatasetGenerator(
        seq_generator, timestep,
        dataset_size=config['dataset_size'],
        output_dir=dataset_root,
        config_file_path=config['config_file_path'],
        generate_train=False,
        generate_test=True,
    )
    
    dataset_generator.generate()