import os
import argparse
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from data import load_data
from model import get_model
from train import train_step, evaluate, LRScheduler, BetaScheduler, configure_experiment


### ENVIRONMENTAL SETTINGS
# to prevent over-threading
torch.set_num_threads(1)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', choices=['synthetic', 'synthetic_noised', 'synthetic_tasknoised', 'celeba'])
parser.add_argument('--model', type=str, default='mtp', choices=['stp', 'jtp', 'mtp'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')

parser.add_argument('--n_steps', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--global_batch_size', type=int, default=-1)
parser.add_argument('--epsilon', type=float, default=-1)
parser.add_argument('--module_sizes', '-ms', nargs='+', default=[])
parser.add_argument('--M_range', '-mr', nargs='+', default=[])
parser.add_argument('--no_stochastic_path', '-nsp', default=False, action='store_true')
parser.add_argument('--no_deterministic_path', '-ndp', default=False, action='store_true')
parser.add_argument('--implicit_global_latent', '-igl', default=False, action='store_true')
parser.add_argument('--global_latent_only', '-glo', default=False, action='store_true')
parser.add_argument('--cnp_det', default=False, action='store_true')
parser.add_argument('--cnp_stc', default=False, action='store_true')

args = parser.parse_args()

# load config
with open(os.path.join('configs', args.data, 'config_{}.yaml'.format(args.model))) as f:
    config = EasyDict(yaml.safe_load(f))

# configure logging and checkpointing paths
logger, save_path, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)

# set device
device = torch.device('cuda')

# load train and valid data
train_iterator, valid_loader = load_data(config, device, split='trainval')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler = LRScheduler(optimizer, config.lr_schedule, config.lr, config.n_steps, config.warmup_iters)
beta_scheduler = BetaScheduler(config.beta_schedule, config.beta, config.n_steps)

# load pretrained model as an imputer if needed
if config.imputer_path != '' and os.path.exists(config.imputer_path):
    ckpt_imputer = torch.load(config.imputer_path)
    config_imputer = ckpt_imputer['config']
    imputer = get_model(config_imputer, device)
    imputer.load_state_dict(ckpt_imputer['model'])
else:
    imputer = config_imputer = None


### MAIN LOOP
best_error = float('inf')
pbar = tqdm.tqdm(total=config.n_steps, initial=0,
                 bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
while logger.global_step < config.n_steps:
    # train step
    train_data = next(train_iterator)
    train_step(model, optimizer, config, logger, *train_data)
    lr_current = lr_scheduler.step()
    config.beta = beta_scheduler.step()
    
    # logging
    if logger.global_step % config.log_iter == 0:
        logger.log_values(log_keys, pbar, 'train', logger.global_step)
        logger.reset(log_keys)
        logger.writer.add_scalar('train/lr', lr_current, logger.global_step)
        logger.writer.add_scalar('train/beta', config.beta, logger.global_step)

    # evaluate and visualize
    if logger.global_step % config.val_iter == 0:
        valid_errors = evaluate(model, valid_loader, device, config, logger, imputer, config_imputer, tag='valid')
        curr_error = sum([valid_errors[task] for task in config.tasks])
    
    # save model
    if logger.global_step % config.save_iter == 0:
        if curr_error < best_error:
            best_error = curr_error
            torch.save({'model': model.state_dict(), 'config': config_copy},
                       os.path.join(save_path, 'best.pth'))
    
    pbar.update(1)

### Save Model and Terminate.
torch.save({'model': model.state_dict(), 'config': config_copy},
           os.path.join(save_path, 'last.pth'))
    
pbar.close()
