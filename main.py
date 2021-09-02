import os
import argparse
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from dataset import load_data
from model import get_model, DataParallel
from train import train_step, evaluate, LRScheduler, configure_experiment


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
parser.add_argument('--data', type=str, default='fss1k', choices=['fss1k'])
parser.add_argument('--architecture', type=str, default='acnp', choices=['cnp', 'acnp'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')

# model-specific arguments
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
if torch.cuda.device_count() > 1:
    model = DataParallel(model).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler = LRScheduler(optimizer, config.lr_schedule, config.lr, config.n_steps, config.lr_warmup)

# for best checkpointing
best_miou = 0.0

### MAIN LOOP
pbar = tqdm.tqdm(total=config.n_steps, initial=0,
                 bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
while logger.global_step < config.n_steps:
    # train step
    train_data = next(train_iterator)
    train_step(model, optimizer, config, logger, *train_data)
    
    # schedulers step
    lr_scheduler.step()
    
    # logging
    if logger.global_step % config.log_iter == 0:
        logger.log_values(log_keys, pbar, 'train', logger.global_step)
        logger.reset(log_keys)
        logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)

    # evaluate and visualize
    if logger.global_step % config.val_iter == 0:
        valid_miou = evaluate(model, valid_loader, device, config, logger, tag='valid')
    
    # save model
    if logger.global_step % config.save_iter == 0:
        if valid_miou < best_miou:
            best_miou = valid_miou
            torch.save({'model': model.state_dict(), 'config': config_copy,
                        'miou': valid_miou, 'global_step': logger.global_step},
                        os.path.join(save_dir, 'best.pth'))
                    
    
    pbar.update(1)

### Save Model and Terminate.
torch.save({'model': model.state_dict(), 'config': config_copy,
            'miou': valid_miou, 'global_step': logger.global_step},
            os.path.join(save_dir, 'last.pth'))
    
pbar.close()
