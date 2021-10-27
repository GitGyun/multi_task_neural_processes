import os
import argparse
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from dataset import load_data
from efficientwnet import get_model, get_model2, DataParallel
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
parser.add_argument('--model', type=str, default='efficientnet-b0', choices=['efficientnet-b0'])
parser.add_argument('--data', type=str, default='fss1k', choices=['fss1k'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')
parser.add_argument('--continue_mode', '-ctn', default=False, action='store_true')
parser.add_argument('--log_dir', type=str, default='')

# model arguments
parser.add_argument('--enc_attn', '-eat', type=str2bool, default=False)
parser.add_argument('--dec_attn', '-dat', type=str2bool, default=False)
parser.add_argument('--attn_architecture', '-aa', default=False, action='store_true')
parser.add_argument('--n_attn_layers', '-nal', type=int, default=1)
parser.add_argument('--double_cross', '-dc', type=str2bool, default=False)

# training arguments
parser.add_argument('--n_steps', type=int, default=-1)
parser.add_argument('--global_batch_size', '-gbs', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1.)
parser.add_argument('--lr_schedule', '-lrs', type=str, default='', choices=['constant', 'sqroot', 'cos', 'poly'])
parser.add_argument('--diff_lr', '-dlr', type=str2bool, default=True)
parser.add_argument('--ways', type=int, default=-1)
parser.add_argument('--shots', type=int, default=-1)

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
if config.attn_architecture:
    model = get_model2(config.model, config.double_cross).to(device)
else:
    model = get_model(config.model).to(device)
if torch.cuda.device_count() > 1:
    model = DataParallel(model).to(device)

if args.diff_lr:
    optimizer = torch.optim.Adam([{'params': model.task_parameters()},
                                  {'params': model.domain_parameters(), 'lr': 0.1*config.lr}], lr=config.lr)
else:
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
        if valid_miou > best_miou:
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
