# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

import random
import logging
from dataclasses import dataclass

import numpy as np

from symseqbench.seq_utils.grammar_generation import generate_grammar
from symseqbench.seq_utils.artificial_grammar import ArtificialGrammar

logger = logging.getLogger('generator')


def determine_decimal_digits(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def rotate(l, n):
    return l[n:] + l[:n]

@dataclass
class GeneratorSample:
    class_seq:          list[int]
    state_seq:          list[str]
    length:    int

class SequenceGenerator:

    def __init__(
        self,
        params,
        mode=None,
        compute_te=False,
        plot_transition_table=False
    ):
        self.params = params
        self.seq_len = params['seq_len_max']
        self.mode = mode
        self.seed = params['seed']
        self.compute_te = compute_te
        self.plot_transition_table = plot_transition_table
        self.n_max_tries = 1e4  # number of maximum attempts to generate a string of correct length
        self.num_illustration_seq = 4
        
        if self.seed is None:
            self.rng = np.random.default_rng()
            logger.warning("Results will not be reproducible!")
        else:
            self.rng = np.random.default_rng(seed=self.seed)

        self.combine_sequences = params['combine_sequences']
        self.combined_seq_len = params['combined_seq_length']

        logger.info(f"Task RNG seed {self.seed}")

        #TODO remove vocab_size from dict
        if 'vocab_size' in dict(params['gramm']):
            del params['gramm']['vocab_size']

        grammar = generate_grammar(**params['gramm'], rng=self.rng) 
        self.sequencer = ArtificialGrammar(**grammar, rng=self.rng)

        # compute TE, correct needs to be set to True
        if self.compute_te:
            transition_table = (self.sequencer.transition_table(correct=True,
                                                                display=False) > 0).astype(int)
            TE = self.sequencer.topological_entropy(transitions=transition_table,
                                                    method='direct')
            try:
                import wandb
                wandb.log({'TE': TE})
            except:
                pass

            print(f"TE:\t{TE}")
            print("####")

        # plot transition table
        if self.plot_transition_table:
            import matplotlib.pyplot as plt
            from sel_bench.seq_utils.markov_chain import MarkovChain

            P = self.sequencer.transition_table(correct=False,
                                                display=True).T
            mc = MarkovChain(P, self.sequencer.states,
                             node_fontsize=10,
                             node_radius=1.,
                             fontsize=10)
            #fig = mc.draw(title=f"start states: {grammar['start_states']}, TE: {round(TE, 2)}")

            for _ in range(self.num_illustration_seq):
                x = self.generate_sequence()
                seqs = x.state_seq
                print(seqs)

            start_states = [str(item) for item in grammar['start_states']]
            try:
                mc.draw(title=f"start states: {start_states}, TE: {round(TE, 2)}",
                        figsize=(5,5))
            except:
                mc.draw(title=f"start states: {start_states}",
                        figsize=(5,5))

            fname = "grammar" 
            path = "."
            #TE_r = round(TE, 2)
            print(f'Save {path}/{fname}.pdf and {path}/{fname}.png')
            plt.savefig(f'{path}/{fname}.pdf')
            plt.savefig(f'{path}/{fname}.png', dpi=300)

            plt.close()

        self.inp_tokens = self.sequencer.tokens
        self.out_tokens = self.sequencer.tokens
  
    def generate(self, idx, compute_length=True):
        
        random.seed(self.seed * idx)
        np.random.seed(self.seed * idx)
        self.sequencer.rng = np.random.default_rng(self.seed * idx)

        if self.combine_sequences:
            gen_sample = self.generate_sequences()
        else:
            gen_sample =  self.generate_sequence()

        if compute_length:
            gen_sample.length = self.__compute_sequence_length(gen_sample.class_seq)
        
        return gen_sample

    def __compute_sequence_length(self, seq):
        return len(seq)

    def generate_sequences(self):

        comb_sample = GeneratorSample(
            np.array([]),
            np.array([]),
            0
        )

        seq_len = 0

        while seq_len < self.combined_seq_len:
            sample = self.generate_sequence()
            comb_sample = self.__concat_samples(comb_sample, sample)
            seq_len = comb_sample.class_seq.shape[0]

        comb_sample.class_seq = comb_sample.class_seq[:self.combined_seq_len]
        comb_sample.state_seq = comb_sample.state_seq[:self.combined_seq_len]

        return comb_sample
    
    def __concat_samples(self, comb_sample, sample):
        comb_sample.class_seq = np.concatenate((comb_sample.class_seq, sample.class_seq))
        comb_sample.state_seq = np.concatenate((comb_sample.state_seq, sample.state_seq))

        comb_sample.class_seq = comb_sample.class_seq.astype(int)

        return comb_sample

    def generate_sequence(self):

        sequence = []
        cnt = 0
        while not (self.params['seq_len_min'] <= len(sequence) <= self.params['seq_len_max']):
            sequence = self.sequencer.generate_string(max_length=self.params['seq_len_max'])

            assert isinstance(sequence, tuple)
            
            sequence = sequence[0] 

            if cnt > self.n_max_tries:
                raise RuntimeError("Could not generate a string of wanted length!")
            cnt += 1

        if not self.combine_sequences:
            assert self.__is_valid(sequence)

        state_seq = sequence

        sequence = sequence[:-1] # If A0 B1 C2 # -> A0 B1 C2
        class_indices = [sym[0] if len(sym) > 1 else sym for sym in sequence] # A0 B1 C2 -> A B C
        class_indices = [ord(s)-64 if s.isupper() else ord(s)-70 for s in class_indices] + [0] # A B C -> 1 2 3 0

        assert len(class_indices) == len(state_seq)

        class_indices = np.array(class_indices)
        state_seq = np.array(state_seq)

        return GeneratorSample(
            class_indices,
            state_seq,
            0
        )

    def __is_valid(self, sequence):
        if sequence[-1] != '#':
            return False
        return True 
