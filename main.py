import os
import argparse
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from dataset import load_data
from model import get_model
from train import train_step, evaluate, LRScheduler, HPScheduler, configure_experiment


### ENVIRONMENTAL SETTINGS
# to prevent over-threading
torch.set_num_threads(1)

def str2bool(v):
    if v == 'True' or v == 'true': return True
    elif v == 'False' or v == 'false': return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# argument parser
parser = argparse.ArgumentParser()

# basic arguments
parser.add_argument('--data', type=str, default='synthetic', choices=['template', 'synthetic', 'synthetic_noised', 'synthetic_tasknoised', 'celeba'])
parser.add_argument('--model', type=str, default='mtp', choices=['stp', 'jtp', 'mtp', 'mtp_glo'])
parser.add_argument('--architecture', type=str, default='anp', choices=['np', 'cnp', 'anp', 'acnp', 'dnp'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')

# model-specific arguments
parser.add_argument('--task_embedding', '-te', type=str2bool, default=None)
parser.add_argument('--dim_hidden', type=int, default=-1)
parser.add_argument('--module_sizes', '-ms', nargs='+', default=[])

parser.add_argument('--n_attn_heads', '-nat', type=int, default=-1)
parser.add_argument('--activation', '-act', type=str, default='')
parser.add_argument('--layernorm', '-ln', type=str2bool, default=None)
parser.add_argument('--dropout', '-dr', type=float, default=-1.)
parser.add_argument('--skip', type=str2bool, default=None)

# training arguments
parser.add_argument('--n_steps', type=int, default=-1)
parser.add_argument('--global_batch_size', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1.)
parser.add_argument('--lr_schedule', '-lrs', type=str, default='', choices=['constant', 'sqroot', 'cos', 'poly'])
parser.add_argument('--beta_T_schedule', '-bts', type=str, default='', choices=['constant', 'linear_warmup'])
parser.add_argument('--beta_G_schedule', '-bgs', type=str, default='', choices=['constant', 'linear_warmup'])

args = parser.parse_args()

# load config
with open(os.path.join('configs', args.data, 'config_train.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))

# configure logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)

# set device
device = torch.device('cuda')

# load train and valid data
train_iterator, valid_loader = load_data(config, device, split='trainval')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler = LRScheduler(optimizer, config.lr_schedule, config.lr, config.n_steps, config.lr_warmup)
if config.global_latent:
    beta_G_scheduler = HPScheduler(config, 'beta_G', config.beta_G_schedule, config.beta_G, config.n_steps, config.beta_G_warmup)
if config.task_latents:
    beta_T_scheduler = HPScheduler(config, 'beta_T', config.beta_T_schedule, config.beta_G, config.n_steps, config.beta_T_warmup)

# load pretrained model as an imputer if needed
if config.model == 'jtp' and config.gamma_valid > 0:
    assert os.path.exists(config.imputer_path)
    ckpt_imputer = torch.load(config.imputer_path)
    config_imputer = ckpt_imputer['config']
    imputer = get_model(config_imputer, device)
    imputer.load_state_dict_(ckpt_imputer['model'])
else:
    imputer = config_imputer = None

# for best checkpointing
best_nll = float('inf')
best_nlls = {block: float('inf') for block in [','.join(task_block) for task_block in config.task_blocks]}
best_error = float('inf')
best_errors = {block: float('inf') for block in [','.join(task_block) for task_block in config.task_blocks]}
if config.model == 'stp':
    best_nll_state_dict = model.state_dict_()
    best_error_state_dict = model.state_dict_()
    
### MAIN LOOP
pbar = tqdm.tqdm(total=config.n_steps, initial=0,
                 bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
while logger.global_step < config.n_steps:
    # train step
    train_data = next(train_iterator)
    train_step(model, optimizer, config, logger, *train_data)
    
    # schedulers step
    lr_scheduler.step()
    if config.global_latent:
        beta_G_scheduler.step()
    if config.task_latents:
        beta_T_scheduler.step()
    
    # logging
    if logger.global_step % config.log_iter == 0:
        logger.log_values(log_keys, pbar, 'train', logger.global_step)
        logger.reset(log_keys)
        logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
        if config.global_latent:
            logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
        if config.task_latents:
            logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

    # evaluate and visualize
    if logger.global_step % config.val_iter == 0:
        valid_nlls, valid_errors = evaluate(model, valid_loader, device, config, logger, imputer, config_imputer, tag='valid')
        valid_nll = sum([valid_nlls[task] for task in config.tasks])
        valid_error = sum([valid_errors[task] for task in config.tasks])
    
    # save model
    if logger.global_step % config.save_iter == 0:
        if config.model == 'stp':
            update_nll = False
            update_error = False
            for block in best_nlls:
                valid_nll_block = sum([valid_nlls[task] for task in block.split(',')])
                if valid_nll_block < best_nlls[block]:
                    best_nlls[block] = valid_nll_block
                    best_nll_state_dict[block] = model.state_dict_block(block)
                    update_nll = True
                    
                valid_error_block = sum([valid_errors[task] for task in block.split(',')])
                if valid_error_block < best_errors[block]:
                    best_errors[block] = valid_error_block
                    best_error_state_dict[block] = model.state_dict_block(block)
                    update_error = True
                    
            if update_nll:
                torch.save({'model': best_nll_state_dict, 'config': config_copy,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': logger.global_step},
                            os.path.join(save_dir, 'best_nll.pth'))
            if update_error:
                torch.save({'model': best_error_state_dict, 'config': config_copy,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': logger.global_step},
                            os.path.join(save_dir, 'best_error.pth'))
        else:
            if valid_nll < best_nll:
                best_nll = valid_nll
                torch.save({'model': model.state_dict_(), 'config': config_copy,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': logger.global_step},
                            os.path.join(save_dir, 'best_nll.pth'))
            if valid_error < best_error:
                best_error = valid_error
                torch.save({'model': model.state_dict_(), 'config': config_copy,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': logger.global_step},
                            os.path.join(save_dir, 'best_error.pth'))
                    
    
    pbar.update(1)

### Save Model and Terminate.
torch.save({'model': model.state_dict(), 'config': config_copy,
            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': logger.global_step},
           os.path.join(save_dir, 'last.pth'))
    
pbar.close()
