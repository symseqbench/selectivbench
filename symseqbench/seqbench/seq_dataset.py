# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""algorithmic_seq_to_seq_problem.py: abstract base class for algorithmic, sequential problems"""

import os
import tqdm
import pickle
from math import ceil
from dataclasses import dataclass
from multiprocessing import Pool, Lock, Manager
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from symseqbench.seqbench.utils.config import Config
from symseqbench.seqbench.utils import get_config_hash
from symseqbench.seqbench.dataset import create_base_dataset_from_config
from symseqbench.seqbench.generator import SequenceGenerator, GeneratorSample


def create_seq_dataset_from_config(config, postfix, is_train=True, pad_index=-1):
    """
    Call this function to create a SeqDataset
    Config should be a symseq.Config.
    """
    base_dataset = create_base_dataset_from_config(config, postfix)

    if 'timestep' in dict(config):
        timestep = config['timestep']
    elif config['inp_enc'] in ('shd', 'ssc'):
        timestep = config['max_time']*1000/config['nb_steps']
    else:
        raise ValueError("timestep is not defined")

    config_hash = get_config_hash(config)

    dataset_root = config['dataset_output_dir']
    dataset_root = f'{dataset_root}-{config_hash}'

    return SeqDataset(
        config,
        base_dataset,
        is_train,
        timestep,
        config['dataset_size'],
        config['do_classify', True],
        pad_index=pad_index,
        read_from_file = config['read_from_file', True],
        dataset_root=dataset_root,
        target_prob_generator=config['target_prob_generator']
    )


@dataclass
class Sample:
    class_seq:          np.array # int
    state_seq:          np.array # str
    target_seq:         np.array # int
    target_probs:       np.array # float
    length:             int

@dataclass
class MCState:
    node:   str
    childs: np.array # str
    probs:  np.array # float


class RestrictedTargetProbGenerator:

    def __init__(self):
        self.num_reduced_states = 0
        self.num_unreduced_states = 0
        self.transitions = None
        self.transition_probs = None
        self.red_state_to_id_map = {}
        self.unred_state_to_id_map = {}
        self.id_to_red_state_map = {}
        self.id_to_unred_state_map = {}
        self.reduced_states = []
        self.unreduced_states = []

    def read_transitions_from_file(self, transitions_filepath):
        print(f'Reading transitions from {transitions_filepath}!')
        with open(transitions_filepath, 'r') as file:
            transitions = DatasetGenerator.read_transitions_from_file(file)
        self.__parse(transitions)

    def read_transitions_from_generator(self, generator):
        print(f'Loading transitions from generator!')
        transitions = {}
        for transition in generator.sequencer.transitions:
            s_from = transition[0]
            s_to = transition[1]
            prob = transition[2]
            
            if s_from not in transitions:
                transitions[s_from] = MCState(
                    s_from, np.array([s_to]), np.array([prob])
                )
            else:
                transitions[s_from].childs = np.append(
                    transitions[s_from].childs, s_to
                )
                transitions[s_from].probs = np.append(
                    transitions[s_from].probs, prob
                )
        self.__parse(transitions)
    
    def __parse(self, transitions):
        self.transitions = transitions
        self.unreduced_states = list(transitions.keys())
        self.num_unreduced_states = len(self.unreduced_states)

        self.__parse_states()
        self.__parse_transitions(transitions)

    def __parse_states(self):
        tmp_id_to_state_map = {}
        for s in self.unreduced_states:
            s_red = self.reduce_state(s)
            if s_red not in self.reduced_states:
                s_red_id = ord(s_red)-64 if s_red.isupper() else ord(s_red)-70
                self.reduced_states.append(s_red)
                self.red_state_to_id_map[s_red] = s_red_id
                tmp_id_to_state_map[s_red_id] = s_red

        self.num_reduced_states = len(self.reduced_states)

        tmp_reduced_states_count = [0 for _ in range(self.num_reduced_states)]
        for s in self.unreduced_states:
            s_red = self.reduce_state(s)
            s_red_id = self.red_state_to_id_map[s_red]
            tmp_reduced_states_count[s_red_id-1] += 1

        id_count = 1
        for s_red_id in range(1,self.num_reduced_states+1):
            s_red = tmp_id_to_state_map[s_red_id]
            s_red_count = tmp_reduced_states_count[s_red_id-1]
            if s_red_count == 1:
                self.unred_state_to_id_map[f'{s_red}'] = id_count
                self.unred_state_to_id_map[f'{s_red}0'] = id_count # just in case
                id_count += 1
            else:
                for i in range(s_red_count):
                    self.unred_state_to_id_map[f'{s_red}{i}'] = id_count
                    id_count += 1

        # Account for eos symbol
        self.red_state_to_id_map['#'] = 0
        self.unred_state_to_id_map['#'] = 0
        self.num_reduced_states += 1
        self.num_unreduced_states += 1
        self.reduced_states.append('#')
        self.unreduced_states.append('#')
        self.id_to_red_state_map = {v: k for k, v in self.red_state_to_id_map.items()}
        self.id_to_unred_state_map = {v: k for k, v in self.unred_state_to_id_map.items()}

    def __parse_transitions(self, transitions):
        transition_probs = {}

        for s in self.unreduced_states:
            transition_probs[s] = np.zeros(self.num_reduced_states)

        for state_name, mcstate in transitions.items():
            for i, child_s in enumerate(mcstate.childs):
                child_s = self.reduce_state(child_s)
                child_s = self.red_state_to_id_map[child_s]
                transition_probs[state_name][child_s] += mcstate.probs[i]

        self.transition_probs = transition_probs

    def __call__(self, state_seq):
        assert self.transition_probs is not None

        target_probs = []
        for state in state_seq:
            if state == '#':
                x = np.zeros(self.num_reduced_states)
                x[0] = 1
                target_probs.append(x)
            else:
                transition_prob = self.transition_probs[state]
                target_probs.append(transition_prob)
        return np.array(target_probs)

    def reduce_state(self, s):
        assert len(s) < 4, f'Unreduced states should not have more than 2 characters. Got {s}!'
        return s[0]

    def print_transitions(self):
        for s_from, state in self.transitions.items():
            print(s_from, state)

    def red_state_to_id(self, red_state):
        return self.red_state_to_id_map[red_state]

    def unred_state_to_id(self, unred_state):
        return self.unred_state_to_id_map[unred_state]

    def id_to_red_state(self, id):
        return self.id_to_red_state_map[id]

    def id_to_unred_state(self, id):
        return self.id_to_unred_state_map[id]

    def get_reduced_states_sorted(self):
        return sorted(
                self.reduced_states,
                key = lambda red_state : self.red_state_to_id(red_state)
            )
    
    def get_unreduced_states_sorted(self):
        return sorted(
                self.unreduced_states,
                key = lambda unred_state : self.unred_state_to_id(unred_state)
            )


class SeqDataset(Dataset):

    def __init__(
        self,
        config,
        base_dataset,
        is_train,
        timestep,
        dataset_size,
        do_classify,
        pad_index=-1,
        not_temporal=False,
        base_duration=None,
        load_to_tensor=None,
        transform=None,
        read_from_file=False,
        dataset_root=None,
        target_prob_generator=None
    ):
        self.config = config
        self.base_dataset = base_dataset
        self.is_train = is_train
        self.transform = transform
        self.do_classify = do_classify

        if not base_duration:
            self.base_duration = self.base_dataset[0][0].shape[0]
        else:
            self.base_duration = base_duration

        if load_to_tensor:
            self.load_to_tensor = load_to_tensor
        else:
            self.load_to_tensor = lambda x: x

        if config['seed'] is None:
            self.rng = np.random.default_rng()
            logger.warning("Results will not be reproducible!")
        else:
            self.rng = np.random.default_rng(seed=config['seed'])

        self.pad_index = pad_index
        self.dataset_size = dataset_size
        self.timestep = timestep
        self.not_temporal = not_temporal
        self.base_shape = self.__get_samples_dimensions()
        self.read_from_file = read_from_file
        self.dataset_root = dataset_root

        # We create the target_prob_generator even for classification
        # to get the output_dim from the transitions
        if target_prob_generator == 'restricted': 
            self.target_prob_generator = RestrictedTargetProbGenerator()
        else:
            raise ValueError(f'Did not recognize target_prob_generator {target_prob_generator}!')

        if self.read_from_file:
            if self.__dataset_folder_exists() and not(config['regenerate']):
                self.__prepare_dataset_from_file()
            else:
                assert self.config is not None
                print(f'SeqBench: Did not find dataset in {self.dataset_root}. Creating it first!')
                self.__create_dataset()
                print(f'SeqBench: Created new dataset in {self.dataset_root}!')
                self.__prepare_dataset_from_file()
        else:
            assert self.config is not None
            print('SeqBench: Generating dataset on the fly!')
            self.gs = self.__create_sequence_generator()
            self.target_prob_generator.read_transitions_from_generator(self.gs)

    def __dataset_folder_exists(self):
        if self.dataset_root is None:
            return False
        if not os.path.isdir(self.dataset_root):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, 'config.yaml')):
            return False
        if not os.path.isfile(os.path.join(self.dataset_root, 'transitions')):
            return False
        if self.is_train and not os.path.isfile(os.path.join(self.dataset_root, 'train')):
            return False
        if not self.is_train and not os.path.isfile(os.path.join(self.dataset_root, 'test')):
            return False
        return True
    
    def __create_dataset(self):
        print(f'SeqBench: Generating dataset with config {self.config["config_file_path"]}!')
        self.gs = self.__create_sequence_generator()
        dataset_generator = DatasetGenerator(
            self.gs, self.timestep,
            dataset_size=self.config['dataset_size'],
            output_dir=self.dataset_root,
            config_file_path=self.config['config_file_path'],
            generate_train=self.is_train,
            generate_test=(not self.is_train),
        )
        dataset_generator.generate()

    def __prepare_dataset_from_file(self):
        print(f'SeqBench: Reading dataset from {self.dataset_root}!')
        self.target_prob_generator.read_transitions_from_file(
            os.path.join(self.dataset_root, 'transitions')
        )
        self.buffered_samples = self.__buffer_samples_from_file()

    def __create_sequence_generator(self):
        return SequenceGenerator(self.config)

    @property
    def output_dim(self):
        if self.do_classify:
            return self.target_prob_generator.num_unreduced_states
        else:
            return self.target_prob_generator.num_reduced_states

    def __get_samples_dimensions(self):
        data = self.base_dataset[0][0]
        shape = data.shape
        if self.not_temporal:
            return shape
        else:
            return shape[1:]

    def __buffer_samples_from_file(self):
        if self.is_train:
            data_path = os.path.join(self.dataset_root, 'train')
        else:
            data_path = os.path.join(self.dataset_root, 'test')

        buffered_samples = []
        pbar = tqdm.tqdm(total=self.dataset_size, desc="Buffering samples")
        with open(data_path, 'r') as file:
            for line in file:
                line = line.replace('\n', '').strip()

                gensample = DatasetGenerator.create_gensample_from_str(line)
                sample = self.gensample_to_sample(gensample)

                buffered_samples.append(sample)

                pbar.update(1)

        pbar.close()

        return buffered_samples

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Returns an element of sel_bench. Since we do not need transition probabilities in case of
        classification, a returned element differs between classification and prediction.
        However, the only difference is that no target probabilities are provided
        (and the content of target_seq of course).
        Prediction: [input_seq, target_seq, class_seq, target_prob_seq]
        Classification: [input_seq, target_seq, class_seq], where:
        - input_seq has shape (T, J)
        - target_seq has shape (T, 1)
        - class_seq has shape (T, 1)
        - target_prob_seq has shape (T, C) (C is the number of ambiguous states)
        Note: class_seq is not needed and only provided for debugging.
        """
        assert not(torch.is_tensor(idx)), "idx needs to be an integer"
        
        sample = self.__get_sample(idx)
        
        if self.not_temporal:
            reshape = lambda x : self.load_to_tensor(x).unsqueeze(0)
        else:
            reshape = lambda x : self.load_to_tensor(x)

        if self.do_classify:
            # This is the place holder that should not be used in that context
            target_probs_or_placeholder = sample.class_seq
        else:
            target_probs_or_placeholder = sample.target_probs

        data = []
        target = []
        gap_mask = []
        target_probs = []
        timestamp = 0
        seq_element = 0
        for data_idx, target_idx, state_idx, target_prob in zip(
                sample.class_seq,
                sample.target_seq,
                sample.state_seq,
                target_probs_or_placeholder
            ):
 
            if self.config['inp_enc'] == 'one_hot':
                data_idx = self.base_dataset.class_dict[data_idx]
            else:
                data_idx = int(self.rng.choice(self.base_dataset.class_dict[data_idx])) # 1 2 3 0 -> 283 12002 1233 1737

            cur_data_sample = self.base_dataset[data_idx][0]
            cur_data_sample = reshape(cur_data_sample)

            if seq_element < self.config['gap_start', 0]:
                delay = 0
            else:
                if (self.rng.random() > self.config['gap_prob', 1]):
                    delay = 0
                elif isinstance(self.config['duration'], Config):
                    delay = np.round(self.config['duration']['dist'](**self.config['duration']['params']),
                                    decimals=1)
                else:
                    delay = self.config['duration']

            if self.config['add_nongramm_gap', False]:
                preds_idx = np.where(self.target_prob_generator.transition_probs[state_idx] == 0)[0]
                delay_dur = 0
                n_delays = round(delay/(self.base_duration*self.timestep))
                if len(preds_idx) != 0:
                    delays = []
                    for i in range(n_delays):
                        idx = np.random.choice(preds_idx)

                        if self.config['inp_enc'] == 'one_hot':
                            delay_idx = self.base_dataset.class_dict[idx]

                        else:
                            delay_idx = int(self.rng.choice(self.base_dataset.class_dict[idx]))
                        
                        delay_content = self.base_dataset[delay_idx][0]
                        delay_content = reshape(delay_content)
                        delays.append(delay_content)
                        delay_dur += delay_content.shape[0] #builds effective delay_dur progressively
                            
                    data.append(cur_data_sample)
                    data += delays
                else:
                    delay_content = torch.zeros([delay_dur] + list(self.base_shape))
                    
                    data.append(cur_data_sample)
                    data.append(delay_content)
            else:
                delay_dur = ceil(delay/self.timestep)
                delay_content = torch.zeros([delay_dur] + list(self.base_shape))
                 
                data.append(cur_data_sample)
                data.append(delay_content)
                    
            data_t = cur_data_sample.shape[0]

            #data.append(cur_data_sample)
            #data.append(delay_content)

            target.append(torch.ones(data_t+delay_dur) * target_idx)

            if not self.do_classify:
                target_prob = torch.tensor(target_prob).unsqueeze(0)
                target_probs.append(target_prob.repeat(data_t+delay_dur, 1))

            if self.do_classify:
                gap_mask.append(
                   torch.cat([torch.zeros(timestamp+data_t-1), torch.ones(delay_dur+1)])
                )
            else:
                gap_mask.append(torch.zeros(data_t-1))
                gap_mask.append(torch.ones(delay_dur+1))
            
            timestamp = timestamp + data_t + delay_dur
            seq_element += 1

        data = torch.cat(data, dim=0)

        if self.config['noise', None]:
            data = data.float()
            rand_noise = torch.rand_like(data)*self.config['noise', 0]
            data += rand_noise

        target = torch.cat(target, dim=0)
        if not self.do_classify:
            target_probs = torch.cat(target_probs, dim=0)

        assert data.shape[0] == target.shape[0]

        if self.do_classify:
            gap_mask = torch.nn.utils.rnn.pad_sequence(gap_mask, batch_first=True)
        else:
            gap_mask = torch.cat(gap_mask, dim=0)
        
        if self.do_classify:
            assert data.shape[0] == target.shape[0]
            sample = [data, target, sample.class_seq, gap_mask]
        else:
            assert data.shape[0] == target.shape[0] == target_probs.shape[0]
            class_seq = [self.target_prob_generator.unred_state_to_id(s) for s in sample.state_seq]
            sample = [data, target, class_seq, target_probs, gap_mask]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __get_sample(self, idx):
        if self.read_from_file:
            return self.__get_sample_from_buffer(idx)
        else:
            gensample = self.gs.generate(idx, compute_length=False)
            return self.gensample_to_sample(gensample)

    def __get_sample_from_buffer(self, idx):
        return self.buffered_samples[idx]

    def gensample_to_sample(self, gensample):
        target_seq = self.create_target_for_gensample(gensample, self.do_classify)
        if self.do_classify:
            target_probs = None
        else:
            target_probs = self.target_prob_generator(gensample.state_seq)
        return Sample(
            gensample.class_seq,
            gensample.state_seq,
            target_seq,
            target_probs,
            gensample.length,
        )

    def create_target_for_gensample(self, gensample, do_classify):
        if do_classify:
            return np.array(
                [self.target_prob_generator.unred_state_to_id(s) for s in gensample.state_seq]
            )
        else:
            # temporary fix till we have a better solution  
            # it could be then we need to have the target also generated by the generator
            if gensample.class_seq[-1] == 0:
                cls_seq = gensample.class_seq
                cls_seq[np.where(cls_seq==0)[0][:-1]+1] = 0
                return np.append(cls_seq[1:], [0])
            else:
                cls_seq = gensample.class_seq
                cls_seq[np.where(cls_seq==0)[0]+1] = 0
                return np.append(cls_seq[1:], [0])
            #return np.append(gensample.class_seq[1:], [0])

class DatasetGenerator:

    def __init__(
        self,
        seq_generator,
        timestep,
        dataset_size=0,
        output_dir=None,
        config_file_path=None,
        generate_train=False,
        generate_test=False,
        append_hash_to_output_dir=False
    ):
        self.seq_generator = seq_generator
        self.timestep = timestep
        self.dataset_size = dataset_size
        self.output_dir = output_dir
        self.config_file_path = config_file_path
        self.generate_train = generate_train
        self.generate_test = generate_test
    
    @staticmethod
    def create_gensample_from_str(line):
        line = line.split('::')
        
        #input_seq = line[0].replace('[', '').replace(']', '')
        #input_seq = input_seq.split(',')
        #input_seq = [int(d.strip()) for d in input_seq]
        
        class_seq = line[0].replace('[', '').replace(']', '')
        class_seq = class_seq.split(',')
        class_seq = [int(d.strip()) for d in class_seq]

        state_seq = line[1].replace('[', '').replace(']', '')
        state_seq = state_seq.replace("'", '')
        state_seq = state_seq.split(',')
        state_seq = [d.strip() for d in state_seq]

        #print(class_seq, state_seq)
        #print(len(class_seq), len(state_seq))

        assert len(class_seq) == len(state_seq)

        length = int(line[2])

        return GeneratorSample(
            np.array(class_seq),
            np.array(state_seq),
            length
        )

    @staticmethod
    def read_transitions_from_file(file):
        transitions = {}
        for line in file:
            line = line.replace('\n', '').strip()
            line = line.replace('(', '').replace(')', '')
            line = line.replace("'", '')
            line = line.split(',')
            s_from = line[0].strip()
            s_to = line[1].strip()
            prob = float(line[2].strip())

            if s_from not in transitions:
                transitions[s_from] = MCState(
                    s_from, np.array([s_to]), np.array([prob])
                )
            else:
                transitions[s_from].childs = np.append(
                    transitions[s_from].childs, s_to
                )
                transitions[s_from].probs = np.append(
                    transitions[s_from].probs, prob
                )
        return transitions

    @staticmethod
    def write_gensample_to_file(file, gensample):
        class_seq = gensample.class_seq.tolist()
        state_seq = gensample.state_seq.tolist()

        assert len(class_seq) == len(state_seq)

        file.write(
            f'{class_seq}::{state_seq}::{gensample.length}\n'
        )

    def generate(self):
        os.makedirs(self.output_dir, exist_ok=True)

        print(f'Generating in {self.output_dir}!')
        if self.generate_train:
            self.__generate_for_dataset_parallelize('train', self.dataset_size)
            #self.__generate_for_dataset('train', self.dataset_size)

        if self.generate_test:
            self.__generate_for_dataset_parallelize('test', self.dataset_size)
            #self.__generate_for_dataset('test', self.dataset_size)

        self.__write_config_to_file()
        self.__write_transitions_to_file()

    def _generate_single(self, args):
        idx, generator, output_file, lock = args
        gensample = generator.generate(idx)
        # Use lock to safely write to shared file
        with lock:
            with open(output_file, 'a') as f:
                DatasetGenerator.write_gensample_to_file(f, gensample)

    def __generate_for_dataset(self, filename, dataset_size):
        print(f'Generating {filename}!')
        file = open(os.path.join(self.output_dir, filename), 'w')
        for i in tqdm.tqdm(range(dataset_size)):
            gensample = self.seq_generator.generate(i)
            DatasetGenerator.write_gensample_to_file(file, gensample)
        file.close()

    # it fails in certain cases. Needs to be checled thoroughly
    def __generate_for_dataset_parallelize(self, filename, dataset_size):
        print(f'Generating {filename}!')

        output_file = os.path.join(self.output_dir, filename)
        
        # Create a manager to share the lock between processes
        with Manager() as manager:
            lock = manager.Lock()
            
            # Clear the file before starting
            with open(output_file, 'w') as f:
                pass
            
            # Create argument tuples for each task
            args_list = [(i, self.seq_generator, output_file, lock) 
                         for i in range(dataset_size)]

            # Use multiprocessing to generate and write samples in parallel
            with Pool() as pool:
                list(tqdm.tqdm(
                    pool.imap(self._generate_single, args_list),
                    total=dataset_size,
                    desc="Generating samples"
                ))

    def __write_config_to_file(self):
        import yaml

        print(f'Writing config!')

        if self.config_file_path is None:
            config = {}
        else:
            with open(self.config_file_path, 'r') as f:    
                config = yaml.safe_load(f)

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as file:
            file.write(yaml.safe_dump(config))

    def __write_transitions_to_file(self):
        print(f'Writing transitions!')
        with open(os.path.join(self.output_dir, 'transitions'), 'w') as file:   
            for transition in self.seq_generator.sequencer.transitions:
                file.write(f'{transition[0]}, {transition[1]}, {float(transition[2])}\n')


class PadSequence:

    def __init__(self,
        do_classify=None,
        pad_index=-1,
        debug_class=True
    ):
        self.pad_index = pad_index
        self.debug_class = debug_class
        if do_classify:
            self.pad_collate_fn = self.pad_collate_classify
        else:
            self.pad_collate_fn = self.pad_collate_predict

    def __call__(self, batch):
        return self.pad_collate_fn(batch)
    
    def pad_collate_classify(self, batch):
        data = []
        target = []
        mask = []
        lens = []
        gap_masks = []
        debug_class_seq = []

        for data_b, target_b, class_seq_b, gap_mask_b in batch:
            l = data_b.shape[0]

            data.append(data_b)
            target.append(target_b)
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(torch.Tensor(gap_mask_b))
            debug_class_seq.append(class_seq_b)

        max_len = max(tensor.size(1) for tensor in gap_masks)
        max_elem = max(tensor.size(0) for tensor in gap_masks)
        padded_gaps = [F.pad(tensor, (0, max_len - tensor.size(1), 0, max_elem - tensor.size(0))) for tensor in gap_masks]
        gap_mask = torch.stack(padded_gaps)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        target = torch.nn.utils.rnn.pad_sequence(target, 
            batch_first=True, 
            padding_value=self.pad_index
        )
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)

        if self.debug_class:
            debug_class_seq = torch.Tensor(np.array(debug_class_seq))
            debug_class_seq = torch.nn.utils.rnn.pad_sequence(debug_class_seq, batch_first=True)

            return {
                'data': data.float(),
                'labels': target.long(),
                'mask': mask.float(),
                'lens': lens,
                'gap_mask': gap_mask.contiguous(),
                'debug_class_seq': debug_class_seq
            }
        else:
            return {
                'data': data.float(),
                'labels': target.long(),
                'mask': mask.float(),
                'lens': lens,
                'gap_mask': gap_mask.contiguous()
            }


    def pad_collate_predict(self, batch):
        data = []
        target = []
        target_probs = []
        mask = []
        lens = []
        gap_masks = []
        debug_class_seq = []

        for data_b, target_b, class_seq_b, target_probs_b, gap_mask_b in batch:
            l = data_b.shape[0]

            data.append(data_b)
            target.append(target_b)
            target_probs.append(target_probs_b)
            mask.append(torch.ones(l))
            lens.append(l)
            gap_masks.append(gap_mask_b)
            debug_class_seq.append(class_seq_b)

        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        target = torch.nn.utils.rnn.pad_sequence(target, 
            batch_first=True, 
            padding_value=self.pad_index
        )
        target_probs = torch.nn.utils.rnn.pad_sequence(target_probs, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
        lens = torch.as_tensor(lens)
        gap_masks = torch.nn.utils.rnn.pad_sequence(gap_masks, batch_first=True)
        debug_class_seq = torch.Tensor(np.array(debug_class_seq))
        debug_class_seq = torch.nn.utils.rnn.pad_sequence(debug_class_seq, batch_first=True)

        return {
            'data': data.float(),
            'labels': target.long(),
            'target_probs': target_probs.float(),
            'mask': mask.float(),
            'lens': lens,
            'gap_mask': gap_masks,
            'debug_class_seq': debug_class_seq
        }
