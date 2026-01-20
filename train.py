# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2025-present, SelectivBench Contributors

"""
Training script for SelectivBench experiments.
"""

import argparse
import copy
import hashlib
import os
import random
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from symseqbench.seqbench import create_seq_dataset_from_config
from symseqbench.seqbench.seq_dataset import PadSequence
from symseqbench.seqbench.utils import prepare_config
from symseqbench.seqbench.utils.config import Config

from models.ssm_models import CustomModel, RecNet
from utils import (
    get_num_heads,
    seq_classification_acc,
    seq_classification_loss,
    str2bool,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")
    
    parser.add_argument("--model", nargs="+", type=str, default=["mamba"],  help="Name of model to load")
    parser.add_argument("--layers", nargs="+", type=int, default=[4], help="Number of hidden layers in model")
    parser.add_argument("--feat_size", nargs="+", type=int, default=[256], help="List of hidden layers dimensions")
    parser.add_argument("--batch_size", default=64, type=int, help="Testing gap durations")
    parser.add_argument("--gpu_device", type=int, help="Testing gap durations")
    parser.add_argument("--hash_save",  action='store_true', help="Use hash to save models")
    parser.add_argument("--save_plots", action='store_true', help="Save plots for these runs")
    parser.add_argument("--visualize_layers", action='store_true', help="Save layers activity")
    parser.add_argument("--save_model",  default=True, type=str2bool, help="Save model for these runs")
    parser.add_argument("--test_pretrained", action='store_true', help="Only test pre-trained models")
    parser.add_argument("--continued_training", action='store_true', help="Continue training from checkpoint")
    parser.add_argument("--debug", action='store_true',  help="No wandb if debug")
    parser.add_argument("--run_main", action='store_true', help="run the main function without wandb agent if wandb sweep id is not defined")
    parser.add_argument("--pt_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--seed", nargs="+", type=int,  default=[42, 123, 456], help="Random Seed")
    parser.add_argument("--task_seed", nargs="+", type=int,  default=[42], help="Random Seed")
    parser.add_argument("--same_seed", default=True, type=str2bool, help="Use same seed for model and task (based on seed)")
    parser.add_argument("--load_model",  nargs="+", type=str, help="Name of model to load")

    # Optimization parameters
    parser.add_argument("--early_stop", action='store_true', help="Turn on early stopping")
    parser.add_argument("--patience", type=int, default=None, help="Patience iterations for early stopping")
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.2], help="Dropout rate")
    parser.add_argument("--lr", type=float, nargs="+", default=[0.001], help="Learning rate")
    parser.add_argument("--wd", type=float, nargs="+", default=[0.], help="Weight decay")
    parser.add_argument("--grad_acc", default=0, type=int, help="Number of gradient accumulation steps for OOM models")
    parser.add_argument("--optimizer", default='adam', type=str, help="Optimizer type")
    parser.add_argument("--scheduler", default='cosine_schedule_with_warmup', type=str, help="Scheduler type")

    # Sweep configuration
    parser.add_argument("--sweep_id", default=None, type=str, help="Id to repeat sweeps")
    parser.add_argument("--sweep_name", type=str, default='Sweep',  help="Name of the sweep")

    # Dataset parameters
    parser.add_argument("--base_set", type=str, default="gsc", help="Name of base dataset to use")
    parser.add_argument("--classify", action='store_true',  help="Use classification version of dataset")
    parser.add_argument("--dataset_size", default=200000, type=int, help="dataset size")
    parser.add_argument("--test_size", default=1000, type=int, help="test dataset size")
    parser.add_argument("--train_gap_min",default=20, type=float, help="Min and Max training gap durations")
    parser.add_argument("--train_gap_max", default=300, type=float, help="Min and Max training gap durations")
    parser.add_argument("--test_dur", default=None, nargs="+", type=float, help="Testing gap durations")
    parser.add_argument("--train_seqlen_min", default=1, type=int, help="Min and Max training gap durations")
    parser.add_argument("--train_seqlen_max", nargs="+", default=[30], type=int, help="Min and Max training gap durations")
    parser.add_argument("--test_seqlen_min", default=1, type=int, help="Min and Max training gap durations")
    parser.add_argument("--test_seqlen_max", nargs="+", default=[30], type=int, help="Min and Max training gap durations")
    parser.add_argument("--dist_dur", type=str, default='uniform',  help="Name of duration distribution")
    parser.add_argument("--test_noise", nargs="+", type=float, default=0, help="Noise level during test")
    parser.add_argument("--train_noise", nargs="+", type=float, default=0, help="Noise level during training")
    parser.add_argument("--amb", nargs="+", default=7, type=int, help="Number of ambiguities")
    parser.add_argument("--amb_depth", nargs="+", default=16, type=int, help="Depth of ambiguities")  
    parser.add_argument("--test_bs", nargs="+", default=[50], type=int, help="Batch size during testing")
    parser.add_argument("--gap_kldiv", default=True, type=str2bool, help="Compute KLdiv only on gaps")
    parser.add_argument("--vocab", nargs="+", type=int, default=[16],  help="Number of hidden layers in model")
    parser.add_argument("--grammar_gap", default=False, type=str2bool, help="Use mistake grammar elements for gaps")
    parser.add_argument("--grammar_scale", default=False, type=str2bool, help="Use same vocab size and amb depth as number of ambiguities")
    parser.add_argument("--read_from_file", default=True, type=str2bool, help="Read dataset from file instead of generating on the fly")
    parser.add_argument("--only_createset", action='store_true', help="Only create dataset without training")
    parser.add_argument("--gap_start",  nargs="+", type=int, default=[0], help="Number of elements before first gap appears")
    parser.add_argument("--gap_prob",  nargs="+", type=float, default=[1], help="Probability of gap insertion")
    parser.add_argument("--only_createsweep", action='store_true', help="Only create sweep without running it")

    # Mamba-specific parameters
    parser.add_argument("--d_state", nargs="+", type=int, default=[64], help="List of mamba state dimensions")
    parser.add_argument("--dt_rank", nargs="+", type=int, default=[0],  help="rank of dt for Mamba")
    parser.add_argument("--taylor_order", nargs="+", type=int, default=[None],  help="The order of the taylor approximation for the exp function. Used only in a specific version of mamba")
    parser.add_argument("--dt_relu", nargs="+", type=str, default=["soft"], help="Use ReLU or MLP for dt")

    # Attention/Transformer-specific parameters
    parser.add_argument("--att_dropout", nargs="+", type=str2bool, default=[False],  help="Use of positional encoding")
    parser.add_argument("--pos_encoding", nargs="+", type=str2bool, default=[False], help="Use of positional encoding")
    parser.add_argument("--slide_window", nargs="+", type=int, default=[0],  help="Transformer Sliding window")
    parser.add_argument("--warmup_steps", nargs="+", type=int, default=[None],  help="Warmup steps for the scheduler")
    parser.add_argument("--n_head", nargs="+", type=int, default=[4], help="Transformer Sliding window")
    parser.add_argument("--rope_base", nargs="+", default=1000, type=int, help="Base value for RoPE positional encoding")

    # Activation function used to create a neuromorphic version of mamba and s4
    parser.add_argument("--act_function", type=str, default='none', help="Activation function type")

    args = parser.parse_args()

    return args


def test(config, test_model, test_size, set, num_classes, dtype, wandb_log="test", wandb_commit=True):
    """
    Automated test function
    """
    test_model.eval()
    eps = 0.000001
    test_BS = config['test_bs']

    test_loader = torch.utils.data.DataLoader(set,
                                              batch_size=test_BS,
                                              shuffle=True,
                                              collate_fn=PadSequence(
                                                        do_classify=config['classify'],
                                                        pad_index=0
                                                        ),
                                             num_workers=NUM_WORKERS)
    with torch.no_grad():
        test_loss = 0
        test_kldiv = 0
        test_acc = 0

        for step, data in tqdm(enumerate(test_loader), total=len(set) // test_BS):

            inputs = data['data'].to(device=device, dtype=dtype)
            loss_targ = data['labels'].to(device=device)
            if config['classify']:
                target = data['labels'].to(device=device)
            else:
                target = data['target_probs'].to(device=device)
            mask = data['mask'].to(device=device)
            gap_mask = data['gap_mask'].to(device=device)

            if 'debug_class_seq' in data:
                class_seq = data['debug_class_seq']
            else:
                class_seq = None

            if config['model'] == 'gla_layered':
                output = test_model(inputs).logits
            elif config['model'] == 'fox_layered':
                output = test_model(inputs).last_hidden_state
            elif config['model'] in ('delta_net_layered', 'gated_delta_net_layered'):
                output = test_model(inputs).logits.to(torch.float32)
            elif config['model'] in ('delta_net', 'gated_delta_net', 'gated_delta_product', 'mesa_net'):
                output = test_model(inputs).to(torch.float32)
            else:
                output = test_model(inputs)

            logits = output.squeeze()

            if config['classify']:
                loss = seq_classification_loss(logits, loss_targ, mask, F.cross_entropy)
            else:
                if config['gap_kldiv']:
                    mask_bool = gap_mask.type(torch.bool)
                else:
                    mask_bool = mask.type(torch.bool)
                loss_targ_masked = loss_targ[mask_bool]
                loss = F.cross_entropy(logits[mask_bool], loss_targ_masked)

            eff_batch_size = inputs.shape[0]

            test_loss += loss*eff_batch_size/test_size

            if config['classify']:
                acc = seq_classification_acc(logits, target, gap_mask, num_classes=num_classes)
                test_acc += acc*eff_batch_size/test_size
            else:
                targets_masked = target[mask_bool] + eps
                logits_masked = F.log_softmax(logits, dim=-1)[mask_bool]
                targets_masked /= torch.sum(targets_masked, dim=-1, keepdim=True)

                kldiv = F.kl_div(logits_masked,
                                targets_masked,
                                reduction='batchmean')

                test_kldiv += kldiv * eff_batch_size / test_size

        if config['save_plots']:  
            if config['classify']:   
                fig, fig_layer, fig_tsne = test_model.plot(inputs,
                                                           loss_targ,
                                                           output,
                                                           mask,
                                                           num_classes,
                                                           class_seq=class_seq)
            else:
                fig, fig_layer, fig_tsne = test_model.plot(inputs,
                                                           target,
                                                           output,
                                                           mask,
                                                           class_seq=class_seq)

            wandb.log({'vfig': wandb.Image(fig)})

            if fig_layer is not None:
                wandb.log({'vfig_layers': wandb.Image(fig_layer)}) 

            if fig_tsne is not None:
                wandb.log({'vfig_layers_tsne': wandb.Image(fig_tsne)}) 

    del set
    del test_loader

    if config['classify']:
        test_perf = test_acc
    else:
        test_perf = test_kldiv

    if not DEBUG:
        wandb.log({wandb_log+"loss":test_loss}, commit=False)
        if config['classify']:
            wandb.log({wandb_log+"acc":test_acc}, commit=wandb_commit)
        else:
            wandb.log({wandb_log+"KL_div":test_kldiv}, commit=wandb_commit)
        
        
    return test_loss, test_perf, logits, target


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())
    
    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    
    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config=None):

    config['n_head'] = get_num_heads(config['feat_size'], max_head_size=128)

    # TODO: remove this once the issue is fixed
    # Transformer models have issues with certain feat_size values in positional embedding
    # Map problematic values to compatible ones
    if config['model'] == 'transformer':
        if config['feat_size'] == 648:
            config['feat_size'] = 768
        elif config['feat_size'] == 1296:
            config['feat_size'] = 1280
        elif config['feat_size'] == 960:
            return

    assert config['test_bs'] > 2, "Test batch size cannot be 2 or less."
    assert config['batch_size'] > 2, "Train batch size cannot be 2 or less."

    # TODO: remove this once the issue is fixed
    if config['train_gap_max'] >= 1 and config['feat_size'] > 648:
        config['grad_acc'] = 16
    else:
        config['grad_acc'] = -1

    if config['grammar_scale']:
        config['vocab'] = config['amb'] + 1
        config['amb_depth'] = config['amb']

    if DEBUG:
        config = config
        seed = config['seed']
    elif config is not None:
        wandb.init(config=config, mode='online')
        seed = wandb.config.seed
    else:
        wandb.init()
        config = wandb.config
        seed = wandb.config.seed

    set_seed(seed)

    BASE_SET = config['base_set']
    inp_enc = BASE_SET
    vocab_size = config['vocab']

    torch.manual_seed(config['seed'])

    dataset_output_dir = 'Data/SeqBench'
    base_dataset_path = 'Data/datasets'

    if not os.path.exists(dataset_output_dir):
        raise ValueError(f"Dataset output directory {dataset_output_dir} does not exist")
    if not os.path.exists(base_dataset_path):
        raise ValueError(f"Base dataset path {base_dataset_path} does not exist")

    # SeqBench task definition
    params = {
        'sequencer_type': 'artificial_grammar',
        'dataset_output_dir': dataset_output_dir,
        'base_dataset_path':  base_dataset_path,
        'read_from_file': config['read_from_file'],
        'config_file_path': None,
        'append_hash_to_output_dir': True,
        'do_classify': config['classify'],
        'target_prob_generator': 'restricted',
        'inp_enc': inp_enc,
        'batch_size': config['batch_size'],
        'vocab_size': vocab_size,
        'alphabet_size': vocab_size,
        'dataset_size': config['dataset_size'],
        'seq_len_min': config['train_seqlen_min'],
        'seq_len_max': config['train_seqlen_max'],
        'combined_seq_length': config['train_seqlen_max'],
        'noise': config['train_noise'],
        'add_nongramm_gap': config['grammar_gap'],
        'combine_sequences': True,
        'regenerate': False,
        'load_data': True,
        'classify': config['classify'],
        'zero_padded_intervals': True,
        'use_train_data': True,
        'seed': config['seed'],
        'amplitude': 10,
        'gap_start' : config['gap_start'],
        'gap_prob' : config['gap_prob'],
    }

    if inp_enc == 'shd':
        # Time step is computed as max_time * 1000 / nb_steps
        params['max_time'] = 1.4
        params['nb_steps'] = 140
        params['num_bins'] = 5
        params['base_dataset_path'] = os.path.join(params['base_dataset_path'], 'SHD')
    elif inp_enc == 'ssc':
        # Time step is computed as max_time * 1000 / nb_steps
        params['max_time'] = 1.4
        params['nb_steps'] = 140
        params['num_bins'] = 5
        params['base_dataset_path'] = os.path.join(params['base_dataset_path'], 'ssc')
    elif inp_enc == 'one_hot':
        params['timestep'] = 0.1
    elif inp_enc == 'gsc':
        params['timestep'] = 10
    else:
        print(f"Base dataset {inp_enc} not defined")
        raise

    if config['dist_dur'] == 'uniform':
        params['duration'] = {'dist': 'uniform',
                              'params': {'low': config['train_gap_min'],
                                         'high': config['train_gap_max']}}
    elif config['dist_dur'] == 'lognormal':
        params['duration'] = {'dist': 'lognormal',
                              'params': {'mean': config['train_gap_min'],
                                         'sigma': config['train_gap_max']}}
    else:
        print("duration distribution" + config['dist_dur'] + "not implemented")
        raise

    params['gramm'] = {
        'alphabet_size': vocab_size,
        'transition_density': 0.4,    # will be divided by ambiguity_depth in the seqbench code as derived parameter
        'ambiguities': config['amb'],
        'ambiguity_depth': config['amb_depth'],
        'initial_states': 4
    }

    if config['same_seed']:
        params['seed'] = config['seed']
    else:
        params['seed'] = config['task_seed']

    if BASE_SET == "shd":
        input_shape = (201, 140)
    elif BASE_SET == "ssc":
        input_shape = (201, 140)
    elif BASE_SET == "gsc":
        input_shape = (101, 13)
    elif BASE_SET == "one_hot":
        input_shape = (101, config['vocab'])
    else:
        print(f"{BASE_SET} is not defined")
        raise

    if config['hash_save']:
        hash_params = copy.deepcopy(params)
        hash_params['model'] = config['model']
        hash_params['inp_size'] = input_shape[-1]
        hash_params['classify'] = config['classify']
        hash_params['lr'] = config['lr']
        hash_params['dropout'] = config['dropout']
        hash_params['layers'] = config['layers']
        hash_params['d_model'] = config['feat_size']
        hash_params['d_state'] = config['d_state']
        hash_params['window_size'] = config['slide_window']
        hash_params['n_head'] = config['n_head']
        hash_params['pos_encoding'] = config['pos_encoding']
        MODEL_FILE_NAME = hashlib.md5(pformat(hash_params).encode('utf-8')).hexdigest()
        MODEL_SAVE_PATH = './model_saves/'
    else:
        MODEL_SAVE_PATH = './model_saves/'
        MODEL_FILE_NAME = "gsc_seqbench_"+config['model']+"_"+str(config['feat_size'])+"_seed"+str(config['seed'])+"_seqlen"+str(config['train_seqlen_max'])
        if config['train_gap_max']!=105:
            MODEL_FILE_NAME += "_traingap"+str(config['train_gap_max'])
        if config['d_state']!=64:
            MODEL_FILE_NAME += "_dstate"+str(config['d_state'])
        if config['layers']!=6:
            MODEL_FILE_NAME += "_layers"+str(config['layers'])
        if config['dt_rank']!=0:
            MODEL_FILE_NAME +="dt_rank"+str(config['dt_rank'])
        if config['dt_relu'] != "soft":
            MODEL_FILE_NAME += "_dtrelu" + str(config['dt_relu'])
        if config['classify']:
            MODEL_FILE_NAME += "_classify"
        else:
            MODEL_FILE_NAME += "_kldiv"

    if config['test_pretrained'] or config['continued_training']:

        if not os.path.exists(MODEL_SAVE_PATH+MODEL_FILE_NAME+'.pth'):
            print(MODEL_SAVE_PATH+MODEL_FILE_NAME+'.pth' + " doesn't exist")
            wandb.finish(exit_code=1)
            return

    params = Config(params)
    params = prepare_config(params)

    test_params = copy.deepcopy(params)
    test_params['dataset_size'] = config['test_size']

    if config['test_dur']==None:
        test_params['duration'] = config['train_gap_max']
    else:
        test_params['duration'] = config['test_dur']

    test_params['seq_len_min'] = config['test_seqlen_min']
    test_params['seq_len_max'] = config['test_seqlen_max']
    test_params['combined_seq_length'] = config['test_seqlen_max']

    if config['same_seed']:
        test_params['seed'] = config['seed']
    else:
        test_params['seed'] = config['task_seed']

    test_set = create_seq_dataset_from_config(test_params,
                                              'test',
                                              is_train=False)
    if config['classify']:
        d_output = test_set.target_prob_generator.num_unreduced_states
    else:
        d_output = vocab_size

    if config['model'].startswith(('delta_net', 'gated_delta', 'gated_delta_product')):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if config['model'] in ('gla_layered', 'delta_net_layered', 'fox_layered', 'gated_delta_net_layered'):
        from fla.models import GLAConfig, DeltaNetConfig, GatedDeltaNetConfig
        from transformers import AutoModelForCausalLM
       
        assert config['feat_size'] % config['n_head'] == 0, "feat_size must be divisible by n_head"
        head_dim = config['feat_size'] // config['n_head']
        
        if config['model'] == 'gla_layered':
            config_gla = GLAConfig(
                    num_hidden_layers=config['layers'],
                    hidden_size=config['feat_size'],
                    head_dim=head_dim,
                    out_features=d_output,
                    vocab_size=d_output)
            model_base = AutoModelForCausalLM.from_config(config_gla)
            model_base.model.embeddings = torch.nn.Identity()

        elif config['model'] == 'delta_net_layered':
            config_delta = DeltaNetConfig(
                    num_hidden_layers=config['layers'],
                    hidden_size=config['feat_size'],
                    head_dim=head_dim,
                    out_features=d_output,
                    vocab_size=d_output)
            model_base = AutoModelForCausalLM.from_config(config_delta)
            model_base.model.embeddings = torch.nn.Identity()

        elif config['model'] == 'gated_delta_net_layered':
            config_gated_delta = GatedDeltaNetConfig(
                    num_hidden_layers=config['layers'],
                    hidden_size=config['feat_size'],
                    head_dim=head_dim,
                    out_features=d_output,
                    vocab_size=d_output)
            model_base = AutoModelForCausalLM.from_config(config_gated_delta)
            model_base.model.embeddings = torch.nn.Identity()

        elif config['model'] == 'fox_layered':
            from fla.models.forgetting_transformer import (ForgettingTransformerConfig,
                                                           ForgettingTransformerForCausalLM,
                                                           ForgettingTransformerModel)
            config_delta = ForgettingTransformerConfig(
                    num_hidden_layers=config['layers'],
                    hidden_size=config['feat_size'],
                    head_dim=head_dim,
                    out_features=d_output,
                    vocab_size=d_output)

            model_base = ForgettingTransformerModel._from_config(config_delta)
            model_base.embeddings = torch.nn.Identity()

        model = CustomModel(model_base,
                           feat_size=input_shape[-1],
                           hidden_dim=config['feat_size']).to(device=device,
                                                              dtype=dtype)

    else:
        model = RecNet(model=config['model'], 
                   inp_size=input_shape[-1],
                   d_output=d_output, 
                   lr=config['lr'], 
                   dropout=config['dropout'], 
                   layers=config['layers'], 
                   d_model=config['feat_size'],
                   d_state=config['d_state'],
                   dt_rank=config['dt_rank'],
                   dt_relu=config['dt_relu'],
                   taylor_order=config['taylor_order'],
                   reg=True, 
                   rope_base=config['rope_base'],
                   window_size=config['slide_window'], 
                   n_head = config['n_head'], 
                   pos_encoding=config['pos_encoding'],
                   visualize_layers=config['visualize_layers'],
                   act_function=config['act_function']).to(device=device,
                                                                   dtype=dtype)
    
    print(model)

    total_params = sum(p.numel() for p in model.parameters())/(10**6)
    print(f"nb params: {total_params}M")

    if not DEBUG:
        wandb.log({'nb_params': total_params})

    if config['test_pretrained'] or config['continued_training']:
 
        if os.path.exists('./model_saves/'+MODEL_FILE_NAME+'.pth'):
            model.load_state_dict(torch.load(MODEL_SAVE_PATH+MODEL_FILE_NAME+'.pth'))
            print("Loaded model:", MODEL_SAVE_PATH+MODEL_FILE_NAME+'.pth')
        else:
            wandb.finish()
            return
          
    EPOCH_SIZE = 5000  # in number of samples

    if config['scheduler'] == 'cosine_schedule_with_warmup':
        def warmup_cosine_lr_scheduler(optimizer, warmup_steps, total_steps, lr_min=None, base_lr=None):
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return float(0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

            return LambdaLR(optimizer, lr_lambda) 
    elif config['scheduler'] == 'cosine_schedule_with_warmup_lr_min':
        def warmup_cosine_lr_scheduler(optimizer, warmup_steps, total_steps, lr_min=0.1, base_lr=3e-4):
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps  # linear warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return float(lr_min / base_lr + (1 - lr_min / base_lr) * cosine_decay)

            return LambdaLR(optimizer, lr_lambda) 
    else:
        raise ValueError(f"config['scheduler'] doesn't exist")

    if config['model']=='s4':
        optimizer, scheduler = setup_optimizer(
            model, lr=config['lr'],
            weight_decay=0.0001,
            epochs=config['dataset_size'] // EPOCH_SIZE
        )
    else:
        lr = config['lr']
        wd = config['wd']
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-15, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        if config['warmup_steps'] is not None:
            scheduler = warmup_cosine_lr_scheduler(optimizer,
                                                   warmup_steps=config['warmup_steps'],
                                                   total_steps=config['dataset_size']//config['batch_size'],
                                                   lr_min=0.1*config['lr'],
                                                   base_lr=config['lr'])
        else:
            scheduler = warmup_cosine_lr_scheduler(optimizer,
                                                   warmup_steps=config['dataset_size']//config['batch_size']*0.05,
                                                   total_steps=config['dataset_size']//config['batch_size'],
                                                   lr_min=0.1*config['lr'],
                                                   base_lr=config['lr'])

    if config['grad_acc'] == -1:
        # Auto-case: automatically determine gradient accumulation steps
        ACC_STEPS = 1
        EFF_BATCH_SIZE = config['batch_size'] // ACC_STEPS
        repeat_run = True
        while repeat_run:  # Repeat run until model fits in GPU memory
            data = []

            assert EFF_BATCH_SIZE > 2, "Effective batch size cannot be 2 or less. Cannot proceed."

            for i in range(EFF_BATCH_SIZE - 1):
                data_b, _ , _ , _ = test_set[i]
                data.append(data_b)

            data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
            data = data.to(device=device, dtype=dtype)

            try:

                if config['model'] in ('delta_net_layered', 'gated_delta_net_layered'):
                    out = model(data)
                    out = out.logits.to(torch.float32).mean()
                else:
                    out = model(data).mean()

                out.backward() # dummy test
                repeat_run = False
                torch.cuda.empty_cache()
                if not DEBUG:
                    wandb.config.update({'grad_acc': ACC_STEPS}, allow_val_change=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if (isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(e).lower()
                or "illegal memory access" in str(e)
                or "cublas_status_alloc_failed" in str(e).lower()
                ):

                    torch.cuda.empty_cache()
                    ACC_STEPS *= 2
                    EFF_BATCH_SIZE = config['batch_size']//ACC_STEPS
                    print(f"Caught OOM error (PyTorch or Triton). Reducing batch size and retrying with grad acc {ACC_STEPS}...")
                else:
                    print(f"An unexpected error occurred: {e}")
                    raise

    elif config['grad_acc'] == 0:
        EFF_BATCH_SIZE = config['batch_size']
        ACC_STEPS = 1
    else:
        EFF_BATCH_SIZE = config['batch_size']//config['grad_acc']
        ACC_STEPS = config['grad_acc']

    if not config['test_pretrained'] or config['continued_training']:

        seq_dataset = create_seq_dataset_from_config(params, 'train')

        seq_loader = torch.utils.data.DataLoader(
            seq_dataset,
            batch_size=EFF_BATCH_SIZE,
            shuffle=False,
            collate_fn=PadSequence(
                do_classify=params['classify'],
                pad_index=0
            ),
        num_workers=NUM_WORKERS
        )

        if config['only_createset']:
            return

        model.train()

        train_loss = []
        train_div = []
        train_acc = []

        if config['early_stop']:
            if config['patience'] is not None:
                patience = config['patience']
            else:
                patience = (len(seq_dataset) // config['batch_size']) * 0.1

            counter_early_stop = 0
            best_val_loss = float('inf')

        optimizer.zero_grad()
        update_steps = 0
        for step, data in tqdm(enumerate(seq_loader), total=len(seq_dataset) // EFF_BATCH_SIZE):

            assert EFF_BATCH_SIZE > 2, "Effective batch size cannot be 2 or less."

            inputs = data['data'].to(device=device, dtype=dtype, non_blocking=True)
            loss_targ = data['labels'].to(device=device, non_blocking=True)
            if config['classify']:
                target = data['labels'].to(device=device, non_blocking=True)
            else:
                target = data['target_probs'].to(device=device, non_blocking=True)
            mask = data['mask'].contiguous().to(device=device, non_blocking=True)
            gap_mask = data['gap_mask'].to(device=device, non_blocking=True)

            # Convert the mask to bool for indexing
            if 'debug_class_seq' in data:
                class_seq = data['debug_class_seq']
            else:
                class_seq = None

            # Convert mask to bool for indexing
            mask_bool = mask.type(torch.bool)

            if config['model'] == 'gla_layered':
                output = model(inputs).logits
            elif config['model'] == 'fox_layered':
                output = model(inputs).last_hidden_state
            elif config['model'] in ('delta_net', 'gated_delta_net', 'gated_delta_product', 'mesa_net'):
                output = model(inputs).to(torch.float32)
            elif config['model'] in ('delta_net_layered', 'gated_delta_net_layered'):
                output = model(inputs).logits.to(torch.float32)
            else:
                output = model(inputs)

            if config['classify']:
                loss = seq_classification_loss(output.squeeze(), loss_targ, mask, F.cross_entropy)
            else:
                loss = F.cross_entropy(output.squeeze()[mask_bool], loss_targ[mask_bool])

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Early stopping: loss contains NaNs or Inf")
                break

            loss.backward()

            if (step+1)%ACC_STEPS == 0:
                update_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                if config['early_stop']:
                    if loss.item() < best_val_loss:
                        best_val_loss = loss.item()
                        counter_early_stop = 0
                        # Save model checkpoint if needed
                    else:
                        counter_early_stop += 1
                        if counter_early_stop >= patience:
                            print("Early stopping triggered")
                            break

                with torch.no_grad():
                    logits = output.squeeze()

                    train_loss.append(loss)
                    if config['classify']:
                        acc = seq_classification_acc(logits, target, gap_mask, num_classes=d_output)
                        train_acc.append(acc)
                    else:
                        if config['gap_kldiv']:
                            perf_mask = gap_mask.type(torch.bool)
                        else:
                            perf_mask = mask_bool
                        targets_masked = target[perf_mask]
                        logits_masked = F.log_softmax(logits, dim=-1)[perf_mask]
                        targets_masked /= torch.sum(targets_masked, dim=-1, keepdim=True)

                        kldiv = F.kl_div(logits_masked,
                                        targets_masked,
                                        reduction='batchmean')

                        eff_batch_size = inputs.shape[0]

                        train_div.append(kldiv)

                    if config['model'] != 's4':
                        scheduler.step()

                if (update_steps) % (EPOCH_SIZE // (EFF_BATCH_SIZE*ACC_STEPS)) == 0:
                    train_loss = float(torch.stack(train_loss).mean())

                    if config['save_plots']:
                        fig, fig_layer, fig_tsne = model.plot(inputs,
                                                              loss_targ,
                                                              output,
                                                              mask,
                                                              num_classes=d_output,
                                                              class_seq=class_seq)
                        wandb.log({'fig': wandb.Image(fig)})

                        if fig_layer is not None:
                            wandb.log({'fig_layers': wandb.Image(fig_layer)}) 

                        if fig_tsne is not None:
                            wandb.log({'fig_layers_tsne': wandb.Image(fig_tsne)}) 

                    if config['classify']:
                        train_acc = float(torch.stack(train_acc).mean())
                        if not DEBUG:
                            wandb.log({"train_loss": train_loss}, commit=False)
                            wandb.log({"train_acc": train_acc}, commit=False)
                    else:
                        train_div = float(torch.stack(train_div).mean())
                        if not DEBUG:
                            wandb.log({"train_loss": train_loss}, commit=False)
                            wandb.log({"train_KL_div": train_div}, commit=False)

                    test_loss, test_perf, test_logits, test_targs = test(config,
                                                                         model,
                                                                         config['test_size'],
                                                                         set=test_set,
                                                                         num_classes=d_output, 
                                                                         dtype=dtype,
                                                                         wandb_log="test_",
                                                                         wandb_commit=True)


                    if config['model'] == 's4':
                        scheduler.step()
                        print(f"Apply scheduler, learning rate {scheduler.get_last_lr()}")
         
                    if config['classify']:
                        print(f"The train acc is {train_acc}")
                    else:  
                        print(f"The train KL_div is {train_div}")
                    print(f"The test acc is {test_perf}")

                    train_loss = []
                    train_div = []
                    train_acc = []

                    if config['save_model']:
                        print("Saved model:", MODEL_SAVE_PATH+MODEL_FILE_NAME+'.pth')
                        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
                        torch.save(model.state_dict(), MODEL_SAVE_PATH+MODEL_FILE_NAME+'.pth')

    else:
        test_loss, test_perf, test_logits, test_targs = test(config,
                                                             model,
                                                             config['test_size'],
                                                             set=test_set,
                                                             num_classes=d_output, 
                                                             dtype=dtype,
                                                             wandb_log="test_",
                                                             wandb_commit=True)
        print(f"The test acc is {test_perf}")


args = parse_arguments()

try:
    device = torch.device(f"cuda:{args.gpu_device}")
except:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIN_MEMORY = device.type == "cuda"

DEBUG = args.debug # not used
NUM_WORKERS = 8

eps = 0.000001

sweep_config = {
    'name': args.sweep_name,
    'method': 'grid',
    'metric': {
        'name': 'test_acc' if args.classify else 'test_KL_div',
        'goal': 'minimize'
    },
    'parameters': {
        'seed': {
            'values': args.seed
        },
    }
}


config = {}
for arg, value in vars(args).items():
    if isinstance(value, list):
        sweep_config['parameters'][arg] = {'values': value}
        config[arg] = value[0]
    else:
        sweep_config['parameters'][arg] = {'values': [value]}
        config[arg] = value

if __name__ == "__main__":

    if (args.sweep_id is None and args.run_main) or DEBUG:
        main(config)
    elif args.sweep_id:
        wandb.agent("team_name/" + args.sweep_id, function=main)
    else:
        sweep_id = wandb.sweep(sweep_config, project="SeqBench_GSC")
        if not args.only_createsweep:
            wandb.agent(sweep_id, function=main)
