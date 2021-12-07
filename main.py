import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from dataset import load_data
from model import get_model, get_imputer
from train import train_step, evaluate, configure_experiment, get_schedulers, Saver


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)

# parse arguments
from argument import args

# load config
with open(os.path.join('configs', args.data, 'config_train.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)

# set device
device = torch.device('cuda')

# load train and valid data
train_iterator, valid_loader = load_data(config, device, split='trainval')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

# load pretrained model as an imputer if needed
imputer, config_imputer = get_imputer(config, device)

# checkpoint saver
saver = Saver(model, save_dir, config_copy)
    
# MAIN LOOP
pbar = tqdm.tqdm(total=config.n_steps, initial=0,
                 bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
while logger.global_step < config.n_steps:
    # train step
    train_data = next(train_iterator)
    train_step(model, optimizer, config, logger, *train_data)
    
    # schedulers step
    lr_scheduler.step()
    if config.model in ['mtp', 'jtp', 'mtp_s']:
        beta_G_scheduler.step()
    if config.model in ['mtp', 'stp', 'mtp_s']:
        beta_T_scheduler.step()
    
    # logging
    if logger.global_step % config.log_iter == 0:
        logger.log_values(log_keys, pbar, 'train', logger.global_step)
        logger.reset(log_keys)
        logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
        if config.model in ['mtp', 'jtp', 'mtp_s']:
            logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
        if config.model in ['mtp', 'stp', 'mtp_s']:
            logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

    # evaluate and visualize
    if logger.global_step % config.val_iter == 0:
        valid_nlls, valid_errors = evaluate(model, valid_loader, device, config, logger, imputer, config_imputer, tag='valid')
        saver.save_best(model, valid_nlls, valid_errors, logger.global_step)
    
    # save model
    if logger.global_step % config.save_iter == 0:
        # save current model
        saver.save(model, valid_nlls, valid_errors, logger.global_step, f'step_{logger.global_step:06d}.pth')
                    
    
    pbar.update(1)

# Save Model and Terminate.
saver.save(model, valid_nlls, valid_errors, logger.global_step, 'last.pth')
    
pbar.close()
