# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SymSeq Contributors

"""
Randomly generate grammars that satisfy certain constraints.
"""

import random
import logging

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

from symseqbench.symseq.markov_chain import MarkovChain


logger = logging.getLogger('generate_grammar')

def Symbols(n=52):
    '''Creation of a dictionary with the compression symbols, restricted
    up to 52 alphabetical characters from 'A' to 'z'
    
    Parameters
    ----------
    n : int
        Number of symbols within the dictionary.
        
    Return
    ------
    dict :
        Dictionary containing the symbols an their numerical code.
    '''

    range_ = 123
    collection = [chr(i) for i in np.arange(range_) if chr(i).isalpha()]
    dict_symbols = {symbol: i for i, symbol in enumerate(collection) if i < n}
    
    return dict_symbols


def generate_grammar(alphabet_size=5, ambiguities=1, ambiguity_depth=3, initial_states=1, transition_density=0.25,
					 assume_equiprobable=True, min_string_length=1, require_exit_from_all=True,
					 allow_unreachable_cycles=True, label="test", rng=None, verbose=True):
	"""
	Generate artificial grammar following the specified constraints and properties.

	Parameters
	----------
	alphabet_size: int
		Size of the alphabet or number of symbols.
	ambiguities: int
	 	How many symbols can be repeated.
	ambiguity_depth: int
	 	Specifies the number of different instances of each repeatable symbol. This is only relevant if ambiguities > 0.
	initial_states: int
		Number of initial states
	transition_density: float
	assume_equiprobable: bool
	min_string_length: int
		Minimum length of valid strings in the grammar.
	require_exit_from_all: bool
		Ensure that there's an exit path from each state.
	allow_unreachable_cycles: bool
		Ensure that each state can be reached from an initial state and has and exit path. This implies
		require_exit_from_all == True.
	label: str
		Grammar label.
	rng: NumPy RandomState object
	verbose: bool

	Returns
	-------
	grammar_dict: dict
		Dictionary with grammar properties, such as states and transition table.

	"""
	if rng is None:
		rng = np.random.default_rng()
		logger.warning("ArtificialGrammar sequences will not be reproducible!")
	random.seed(rng.integers(0, 100).item())

	# some conditions for which we certainly cannot / should not generate valid grammars
	if ambiguities > 0 and ambiguity_depth == 0:
		raise RuntimeError("Could not generate grammar for the specified parameters!")

	n_max_iter = 1e5
	cnt = 0
	alphabet_size -= 1 # in order to count the # symbol as an alphabet
	path_available = None
	while path_available is None and cnt < n_max_iter:
		# 2. Generate grammar parameters
		alphabet = list(Symbols(alphabet_size).keys())
		states = alphabet.copy()
		states_rep = alphabet.copy()

		if ambiguities:
			# which symbols can repeat
			repeatable_symbols = list(random.sample(alphabet, ambiguities))

			# how many repetitions?
			for rep in repeatable_symbols:
				states.remove(rep)
				# replace with indexed states
				new_states = [rep + str(i) for i in range(ambiguity_depth)]
				states.extend(new_states)

				states_rep.remove(rep)
				new_states_rep = [rep for i in range(ambiguity_depth)]
				states_rep.extend(new_states_rep)
		alphabet.append('#')
		states.append('#')
		states_rep.append('#')

		# 3. Generate transition table
		A = np.zeros((len(states), len(states)))

		for i, src in enumerate(states_rep):
			for a in alphabet:
				x = np.where(a == np.array(states_rep))[0]
				y = rng.permutation(len(x))[0]
				index = x[y]
				if rng.random() < (len(x) * transition_density):
				    A[index, i] = 1

		# DEPR: ensure all states can exit (except terminal) -> no all zeros columns
		# ensure no transitions are made from the terminal symbol '#' (set its column to 0)
		A[:, states.index('#')] = np.zeros_like(A[:, states.index('#')])
		# after this step, the matrix A is used as source_id X target_id
		A = A.T

		# assume equiprobable transitions
		if assume_equiprobable:
			A = normalize(A, axis=1, norm='l1')
		else:
			A *= t.todense()

		transitions = []
		for x, y in zip(A.nonzero()[0], A.nonzero()[1]):
			transitions.append((states[x], states[y], A[x, y]))

		# ensure start states can reach terminal (and start is not terminal)
		start_states = rng.choice(states, initial_states, replace=False).tolist()

		G = nx.DiGraph()
		G.add_nodes_from(states)
		G.add_edges_from([(x[0], x[1]) for x in transitions])

		path_available = _check_grammar_validity(G, start_states, min_string_length, require_exit_from_all,
												 allow_unreachable_cycles, verbose)

		# discard grammar if '#' is among the start states
		if '#' in start_states:
			path_available = None
		cnt += 1

	if path_available is None:
		raise RuntimeError("Could not generate grammar for the specified parameters!")

	return {
		'label': label,
		'states': states,
		'alphabet': alphabet,
		'start_states': start_states,
		'terminal_states': ['#'],
		'transitions': transitions,
		'eos': '#'
	}


def _check_grammar_validity(G, start_states, min_string_length, require_exit_from_all, allow_unreachable_cycles,
							verbose):
	"""
	Check the following constraints:
		- for each start state, there's a path of valid length to the terminal state
		- (optional) for all states, there's a path of any length to the terminal state (no absorbing state)
		- (optional) all states are reachable from start states, and from each there's a path to the terminal state

	Parameters
	----------
	G
	start_states
	min_string_length
	require_exit_from_all
	verbose

	Returns
	-------

	"""
	path_available = None
	states = start_states

	# if exit path required, check paths from all states not just the initial ones
	if require_exit_from_all or not allow_unreachable_cycles:
		states = list(G.nodes)

	# verify all states can exit, and each such path is sufficiently long
	for x in states:
		try:
			path_available = nx.shortest_path(G, source=x, target='#')

			# if shortest path from an initial state is below minimum wanted length, invalidate it and continue search
			if x in start_states and len(path_available) - 1 < min_string_length:
				if verbose:
					logger.info(f"Shortest path from start state {x} to end state is below minimum expected length")
				path_available = None
				break
		except:
			if verbose:
				logger.info("No path from start state {} to end state".format(x))
			path_available = None
			break

	# to avoid unreachable cycle, it's enough to ensure all nodes are reachable from the initial state
	if path_available and not allow_unreachable_cycles:
		for tgt in list(G.nodes):
			if tgt not in start_states:
				reachable = False
				for start in start_states:
					try:
						nx.shortest_path(G, source=start, target=tgt)
						reachable = True
					except nx.NetworkXNoPath:
						break
				if not reachable:
					if verbose:
						logger.info("No path from an initial state to {}".format(tgt))
					return None
	return path_available

def draw_graph(g, max_lift=1, save="./last.png"):
	P = g.transition_table(correct=True, display=True)
	mc = MarkovChain(P, g.states, title=r"$P_{s}$")
	mc.draw()

	for lift in range(max_lift):
		frequencies = chunk_transitions(g.generate_sequence(), lift+1, return_labels=True)
		n_frequencies = normalize(frequencies, axis=1, norm='l1')

		mc = MarkovChain(n_frequencies, list(frequencies.columns), title=r"$P_{freq}$")
		label = save.split(".")[-2] + "_lift{}".format(lift) + save.split(".")[-1]
		mc.draw(label)

def draw_single_graph(g, lift=0, ax=None, save=None, display=True, **kwargs):
	"""
	Plots a graph for a single case, specified by the parameter lift.

	Parameters
	----------
	g: ArtificialGrammar
	lift: int
		The lift value for the grammar.
	ax: matplotlib.Axis
	save
	display: bool
	kwargs
	"""
	transition_table = g.transition_table(correct=True, display=display).T  # must transpose here for correct plotting

	if lift == 0:
		mc = MarkovChain(transition_table, g.states, **kwargs)
		mc.draw(g.terminal_symbols, ax=ax, img_path=save)
	else:
		frequencies = chunk_transitions(g.generate_sequence(), lift+1, return_labels=True)
		n_frequencies = normalize(frequencies, axis=1, norm='l1')

		mc = MarkovChain(n_frequencies, list(frequencies.columns), title=r"$P_{freq}$",
						 fontsize=28, node_fontsize=32)
		mc.draw(g.terminal_symbols, ax=ax)
		mc.draw(ax=ax)

