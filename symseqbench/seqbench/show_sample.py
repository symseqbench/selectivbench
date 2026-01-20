# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""sequence_prediction.py"""

import os
import random
import logging
import argparse
import collections

import torch
import numpy as np

from symseqbench.seqbench.utils.config import Config
from symseqbench.seqbench.utils import prepare_config
from symseqbench.seqbench.seq_dataset import PadSequence
from symseqbench.seqbench import create_seq_dataset_from_config

__script_name__ = os.path.basename(__file__)

logger = logging.getLogger('show_sample')

_DataTuple = collections.namedtuple('DataTuple', ('inputs', 'classes'))

class DataTuple(_DataTuple):
    """
    Tuple used by storing batches of data by problems.
    """
    __slots__ = ()

_AlgSeqAuxTuple = collections.namedtuple(
    'AlgSeqAuxTuple', ('seq_length', 'trans_probs', 'sequences', 'debug_class_seq'))

class AlgSeqAuxTuple(_AlgSeqAuxTuple):
    """
    Tuple used by storing batches of data by algorithmic sequential problems.
    Contains three elements:

    - mask that might be used for evaluation of the loss function
    - length of sequence
    - number of subsequences

    """
    __slots__ = ()


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the .yaml which contains all user defined parameters.')

    args = parser.parse_args()

    return vars(args)

def show_sample(
        data_tuple,
        aux_tuple,
        target_prob_generator,
        path=None,
        fname=None,
        fname_ax=None,
        sample_number=1,
        do_classify=True
    ):
    """
    Shows the sample (both input and target sequences) using matplotlib.
    Elementary visualization.

    :param data_tuple: Data tuple.
    :param aux_tuple: Auxiliary tuple.
    :param sample_number: Number of sample in a batch (DEFAULT: 0)

    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['text.usetex'] = False

    if do_classify:
        alphabet = target_prob_generator.get_unreduced_states_sorted()
    else:
        alphabet = target_prob_generator.get_reduced_states_sorted()
    vocab_size = len(alphabet)

    sample_length = aux_tuple.seq_length[sample_number]

    sequence = aux_tuple.debug_class_seq[sample_number, :]
    sequence = sequence.cpu().detach().numpy()

    # Generate "canvas".
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5),
                                   sharex=True, sharey=False,
                                   gridspec_kw={'height_ratios': [10, 10]})
    # Set ticks.
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Set labels.
    ax1.set_ylabel('Inp. ch.')
    ax2.set_title('Target mask')
    ax2.set_title('Targets')
    ax2.set_ylabel('Targ. ch.')
    ax2.set_xlabel('Time steps')
    
    # print data
    print("\ninputs:", data_tuple.inputs[sample_number, :, :])
    print("\ninputs size:", data_tuple.inputs[sample_number, :, :].size())
    print("\nclasses:", data_tuple.classes[sample_number, :])
    print("\nclasses size:", data_tuple.classes[sample_number, :].size())
    print("\nseq_length:", aux_tuple.seq_length)
    print("\nsequences:", aux_tuple.sequences)
    print("\nsequences:", aux_tuple.sequences)
    print("\ndebug sequence:", sequence)

    # show data.
    params = {}

    if do_classify:
        ax1.set_title(f'Inputs={[target_prob_generator.id_to_red_state(s) for s in sequence]}')
    else:
        Inputs_state=[target_prob_generator.id_to_unred_state(s) for s in sequence]
        Inputs = [sym[0] if len(sym) > 1 else sym for sym in Inputs_state]
        ax1.set_title(f'Inputs={Inputs}')

    ax1.imshow(np.transpose(data_tuple.inputs[sample_number, :, :],  [1, 0]),
               #aspect='auto', origin='lower', interpolation='none', **params)
               aspect='auto', origin='lower', **params)
    
    if do_classify:
        classes = torch.nn.functional.one_hot(data_tuple.classes[sample_number,:sample_length], num_classes=vocab_size)
        classes = torch.swapaxes(classes, 0, 1)
        ax2.imshow(classes, aspect='auto', origin='lower', interpolation='none', **params)
    else:
        #classes = torch.nn.functional.one_hot(data_tuple.classes[sample_number,:sample_length], num_classes=vocab_size)
        #classes = torch.swapaxes(classes, 0, 1)
        #ax2.imshow(classes, aspect='auto', origin='lower', **params)

        ax2.imshow(np.transpose(aux_tuple.trans_probs[sample_number, :, :],  [1, 0]),
               aspect='auto', origin='lower', interpolation='none', **params)

    if vocab_size:
        ax2.set_yticks(np.arange(vocab_size), alphabet)

    # Plot!
    fig.tight_layout()

    if fname and path:
        os.makedirs(path, exist_ok=True)
        print(f'Save figure to {path}/{fname}.pdf and {path}/{fname}.png')
        plt.savefig(f'{path}/{fname}.pdf')
        plt.savefig(f'{path}/{fname}.png', dpi=600)

    plt.close()

    ###################
    # Stimulus
    ###################

    # Generate "canvas".
    fig = plt.figure(figsize=(5, 2.5))

    # Set labels.
    plt.ylabel('Inp. ch.')
    plt.xlabel('Time steps')

    # show data.
    params = {}

    sequence = aux_tuple.debug_class_seq[sample_number, :]
    sequence = sequence.cpu().detach().numpy()

    if do_classify:
        plt.title(f'Inputs={[target_prob_generator.id_to_red_state(s) for s in sequence]}')
    else:
        Inputs_state=[target_prob_generator.id_to_unred_state(s) for s in sequence]
        Inputs = [sym[0] if len(sym) > 1 else sym for sym in Inputs_state]
        plt.title(f'Inputs={Inputs}')

    plt.imshow(np.transpose(data_tuple.inputs[sample_number, :, :],  [1, 0]),
               aspect='auto', origin='lower', **params)

    # Plot!
    fig.tight_layout()

    if fname and path:
        os.makedirs(path, exist_ok=True)
        print(f'Save figure to {path}/{fname_ax}.pdf and {path}/{fname_ax}.png')
        plt.savefig(f'{path}/{fname_ax}.pdf')
        plt.savefig(f'{path}/{fname_ax}.png', dpi=600)


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    args = parse_cli_arguments()
    config = Config.parse_config_from_args(args)
    config.print_config()
    
    config['config_file_path'] = args['config']
    config = prepare_config(config)
    seed = config['seed']
    inp_enc = config['inp_enc']
    config['do_classify'] = True
    config['dataset_size'] = config['dataset_size_train']

    random.seed(seed)
    np.random.seed(seed)

    seq_dataset = create_seq_dataset_from_config(config, 'train')

    seq_loader = torch.utils.data.DataLoader(
        seq_dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=PadSequence(
            do_classify=config['do_classify'],
            pad_index=-1
        ),
        num_workers=0
    )

    batch = next(iter(seq_loader))

    data = batch['data']
    labels = batch['labels']
    len_seqs = batch['lens']
    debug_class_seq = batch['debug_class_seq']

    if not config['do_classify']:
        target_probs = batch['target_probs']
    else:
        target_probs = None

    data_tuple = DataTuple(data, labels)
    aux_tuple = AlgSeqAuxTuple(len_seqs, target_probs, labels, debug_class_seq)

    if config['do_classify']:
        fname = f'show_sample_{inp_enc}_classify'
    else:
        fname = f'show_sample_{inp_enc}_prob'

    show_sample(
        data_tuple,
        aux_tuple,
        seq_dataset.target_prob_generator,
        path='img',
        fname=fname,
        fname_ax=f'stimulus_{inp_enc}',
        do_classify=config['do_classify']
    )
