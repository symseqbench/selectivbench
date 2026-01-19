# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SymSeq Contributors

import os
import pickle as pkl
import numpy as np
import copy
from collections import Counter
from math import log
from gzip import compress
from tqdm import tqdm

from symseqbench.utils import metrics

#from fna.tools import utils
import logging

logger = logging.getLogger(__name__)


def empty(signal):
    """
    Evaluate whether a signal is empty
    :param signal:
    :return: bool
    """
    if isinstance(signal, np.ndarray):
        return not bool(signal.size)  # seq.any() # seq.data
    elif isinstance(signal, list) and signal:
        if isiterable(signal):
            result = np.mean([empty(n) for n in list(itertools.chain(signal))])
        else:
            result = np.mean([empty(n) for n in list(itertools.chain(*[signal]))])
        if result == 0. or result == 1.:
            return result.astype(bool)
        else:
            return not result.astype(bool)
    elif isinstance(signal, pd.DataFrame):
        return signal.empty
    else:
        return not signal


# ######################################################################################################################
class SymbolicSequencer(object):
	"""
	Build patterned symbolic sequences.
	Contains the generic constructors to implement structured symbolic sequences
	"""
	# TODO - load_sequence, plots
	def __init__(self, label, set_size=None, alphabet=None, eos=None, rng=None):
		"""
		:param label: [string] label of the current task
		:param set_size: [int] number of unique symbols
		:param alphabet: [list] unique tokens
		:param eos: [string] end-of-sentence marker
		:param rng: [numpy.random] seeded random number generator
		"""
		logger.info("Creating symbolic sequencer")
		self.tokens = []
		self.name = label
		if alphabet is not None:
			self.tokens = alphabet
		else:
			self.tokens = [str(i) for i in range(set_size)]
		self.eos = eos
		self.string_set = []
		self.input_sequence = []

		if rng is None:
			self.rng = np.random.default_rng()
			logger.warning("SymbolicSequencer sequences will not be reproducible!")
		else:
			self.rng = rng

	def generate_string_set(self, set_length, length_range, verbose=True):
		"""
		Generate the complete stringset for the experiment
		:param set_length: total number of strings to generate (or allowed, if limited)
		:param violations: (int=n_strings) introduce syntactic violations in n_strings in the set
		:return:
		"""
		if verbose:
			logger.info('Generating {0!s} strings...'.format(set_length, self.name))
		self.string_set = self.draw_subsequences(n_subseq=set_length, seq=None, length_range=length_range)

	def generate_sequence(self):
		"""
		Generate a complete sequence by concatenating the strings, separated by the eos marker
		:return: input sequence
		"""
		assert not empty(self.string_set), "String set is empty, generate it first"
		self.input_sequence = self._concatenate_stringset(self.string_set, separator=self.eos)
		return self.input_sequence

	@staticmethod
	def _concatenate_stringset(string_set, separator=''):
		"""
		Concatenates a string set (list of strings) into a list of symbols, placing the separator symbol between strings
		:param string_set: list of strings
		:param separator: string symbol separating different strings
		:return:
		"""
		str_set = copy.deepcopy(string_set)
		if separator is not None:
			[n.insert(0, separator) for idx, n in enumerate(list(str_set)) if idx != 0]
		symbol_seq = np.concatenate(list(str_set)).tolist()
		return symbol_seq

	def generate_random_sequence(self, T=0, verbose=True):
		"""
		Randomly draw items from the alphabet
		:return: input sequence
		"""
		if verbose:
			logger.info('Generating a random sequence of length {0!s}, '
			            'from a set of {1!s} symbols'.format(T, len(self.tokens)))
		if T == len(self.tokens):  # draw without repetition (each symbol only once)
			replace = False
		else:
			replace = True
		return list(self.rng.choice(self.tokens, T, replace=replace))

	def draw_subsequences(self, n_subseq, seq=None, length_range=(5, 10), verbose=False):
		"""
		Draw sample sub-sequences from a main sequence or from the stringset
		:param n_subseq: number of subsequences to draw
		:param seq: sequence to draw from (if no stringset is available)
		:param length_range: tuple (min, max) or None (will be the
		:return:
		"""
		if not empty(self.string_set):
			idx = self.rng.integers(0, len(self.string_set), n_subseq)
			out_str = [self.string_set[ii] for ii in idx]
		else:
			if seq is None:
				# generate a small random sequence
				seq = self.generate_random_sequence(T=int(n_subseq*max(length_range)), verbose=verbose)
			# take random chunks
			idx = self.rng.integers(0, len(seq)-n_subseq, n_subseq)
			if length_range[0] == length_range[1]:
				lengths = [length_range[0] for _ in range(n_subseq)]
			else:
				lengths = self.rng.integers(length_range[0], length_range[1], n_subseq)
			out_str = [seq[ii:ii + xx] for ii, xx in zip(idx, lengths)]
		return out_str

	@staticmethod
	def count(sequence, as_freq=False):
		"""
		Computes the frequency of each item in the sequence
		:param sequence: list of tokens
		:param as_freq: bool - return total counts (False) of frequencies
		:return dict: {item: frequency (count/length)}
		"""
		if as_freq:
			return {k: v/len(sequence) for k, v in Counter(sequence).items()}
		else:
			return dict(Counter(sequence))

	@staticmethod
	def most_common(sequence, n, as_freq=False):
		"""
		Return the frequency of the n most common tokens in the sequence
		:param sequence: list of tokens
		:param n: n most common
		:param as_freq: bool - return total counts (False) of frequencies
		:return:
		"""
		ctr = Counter(sequence).most_common(n)
		if as_freq:
			return {k: v/len(sequence) for (k, v) in ctr}
		else:
			return dict(ctr)

	def entropy(self, sequence):
		"""
		Calculate the entropy (bits) of a sequence.
		:param sequence: full symbolic sequence
		"""
		cnt = [self.count(sequence)[i] for i in np.unique(sequence)]
		d = sum(cnt)
		ent = []
		for i in [float(i) / d for i in cnt]:
			# round corner case that would cause math domain error
			if i == 0:
				i = 1
			ent.append(i * log(i, 2))
		return -1 * sum(ent)

	@staticmethod
	def topological_entropy(seq=None, transitions=None, method="lift"):
		"""
		Compute TE of a grammar, using the "lift" method [1] or the "direct" method (subscripted transitions).
		See, e.g.:
		-------------
		[1] - Bollt and Jones (2000)
		[2] - Shiff and Katan (2014)
		:return:
		"""
		logger.info("Computing topological entropy..")
		if method == "lift":
			assert seq is not None, "The lift method is computed on the generated sequences, sequence needs to be " \
			                        "provided"
			if len(seq) <= 1000:
				logger.warning("Sequence is too short, entropy estimates may not be reliable.")
			TE = []
			nn, top_ent = 0, 0
			max_lift = 15
			for nn in range(max_lift):
				M = metrics.chunk_transitions(seq, nn + 1, display=False)
				eigs = np.linalg.eigvals(M)

				max_eig = np.real(np.max(eigs))
				TE.append(np.log(max_eig))
				logger.info("Lift: {0!s}; Entropy: {1!s}".format(nn + 1, TE[-1]))
				if (len(TE) > 1) and (np.round(np.diff(TE), 1)[-1] == 0.):
					top_ent = TE[-2]
					break
				else:
					top_ent = TE[-1]
				if np.isinf(top_ent):
					break
			return nn, top_ent, TE
		else:
			assert np.array_equal(transitions, transitions.astype(bool)), "The direct method is computed on the " \
			                                                              "corrected transition table, " \
			                                                              "binary transition table must be provided"
			eigs = np.linalg.eigvals(transitions)
			max_eig = np.real(np.max(eigs))
			TE = np.log(max_eig)
			return TE

	@staticmethod
	def compressibility(sequence):
		"""
		Determine the compressibility ratio of the input sequence
		:param sequence: symbolic sequence (list of symbols)
		:return:
		"""
		return len(compress(''.join(sequence).encode())) / len(sequence)

	def string_length(self, string_set=None):
		"""
		Compute the length of all strings in the string_set
		:param string_set: list of strings or None
		:return:
		"""
		if string_set is None and not empty(self.string_set):
			string_set = self.string_set
		elif string_set is None:
			string_set = []

		return [len(string) for string in string_set]

	def string_set_complexity(self, string_set=None):
		"""
		Evaluate the complexity of the string set as the pairwise distance between strings
		:param string_set:
		:return:
		"""
		if string_set is None and not empty(self.string_set):
			string_set = self.string_set
		elif string_set is None:
			string_set = []
		logger.info("Evaluating string set complexity...")
		edit_dists = np.zeros((len(string_set), len(string_set)))
		hamming_dists = np.zeros((len(string_set), len(string_set)))

		iu1 = np.tril_indices_from(edit_dists)
		for i, j in tqdm(zip(iu1[0], iu1[1]), desc="Calculating pairwise distances: ", total=len(iu1[0])):
			edit_dists[i, j] = metrics.edit_distance(string_set[i], string_set[j])
			hamming_dists[i, j] = metrics.hamming_distance(string_set[i], string_set[j])

		return {'edit_distance': edit_dists, 'hamming_distance': hamming_dists}

	# TODO plot distributions (string length, token frequency, ... string complexity), recurrence plots