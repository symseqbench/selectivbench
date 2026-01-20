# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import numpy as np

from symseqbench.seqbench.utils.config import Config

def parse_duration(identifier):
    if identifier == 'uniform':
        return np.random.uniform
    elif identifier == 'lognormal':
        return np.random.lognormal
    else:
        raise ValueError('Unknown duration identifier!')

def prepare_config(config):
    config['gramm']['transition_density'] /= config['gramm']['ambiguity_depth']
    config = parse_duration_all(config)
    return config

def parse_duration_all(config):
    if 'dist' in dict(config['duration']):
        config['duration']['dist'] = parse_duration(config['duration']['dist'])
    return config

def get_config_hash(config):
    import hashlib
    from pprint import pformat

    key = {}

    key['seq_len_max'] = config['seq_len_max']
    key['seq_len_min'] = config['seq_len_min']
    key['combined_seq_length'] = config['combined_seq_length']
    key['gramm'] = config['gramm'].asdict()
    key['dataset_size'] = config['dataset_size']
    key['seed'] = config['seed']

    config_hash = hashlib.md5(pformat(key).encode('utf-8')).hexdigest()

    assert config_hash is not None

    return config_hash
