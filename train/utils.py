import os
import sys
import shutil
import random
import io
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from data import colors


def configure_experiment(config, debug_mode=False):
    # set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # for debugging
    if debug_mode:
        config.n_steps = 3
        config.log_iter = 1
        config.val_iter = 1
        config.save_iter = 1
        config.log_dir += '_debugging'

    # set directories
    exp_name = config.model + config.name_postfix
    os.makedirs('experiments', exist_ok=True)
    os.makedirs(os.path.join('experiments', config.log_dir), exist_ok=True)
    os.makedirs(os.path.join('experiments', config.log_dir, exp_name), exist_ok=True)
    log_path = os.path.join('experiments', config.log_dir, exp_name, 'logs')
    save_path = os.path.join('experiments', config.log_dir, exp_name, 'checkpoints')
    if os.path.exists(save_path):
        if debug_mode:
            shutil.rmtree(save_path)
        else:
            while True:
                print('redundant experiment name! remove existing checkpoints? (y/n)')
                inp = input()
                if inp == 'y':
                    shutil.rmtree(save_path)
                    break
                elif inp == 'n':
                    print('quit')
                    sys.exit()
                else:
                    print('invalid input')
    os.makedirs(save_path)

    # tensorboard logger
    logger = Logger(log_path)
    log_keys = ['nll_{}'.format(task) for task in config.tasks]
    if config.model != 'stp':
        log_keys.append('kld_G')
    if config.model != 'jtp':
        log_keys += ['kld_{}'.format(task) for task in config.tasks]
    for log_key in log_keys:
        logger.register_key(log_key)
    
    return logger, save_path, log_keys


def plot_curves(X_C, Y_C, X_D, Y_D, Y_D_pmeans, logger, Y_C_imp=None):
    '''
    Plot multiple predictions and predictive mean over GT task functions, with imputed values if given.
    '''
    toten = ToTensor()
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    for task in Y_C:
        plt.figure(figsize=(20, 3*(X_C.size(0)//5)))
        for j in range(X_C.size(0)):
            plt.subplot(X_C.size(0)//5, 5, j + 1)
            index = ~Y_C[task][j].isnan()
            plt.scatter(X_C[j, index].cpu(), Y_C[task][j, index].cpu(), color=colors[task], s=15)
            if Y_C_imp is not None:
                plt.scatter(X_C[j, ~index].cpu(), Y_C_imp[task][j, ~index].cpu(), color=colors[task], s=15, marker='x')
            for sample in Y_D_pmeans[task][j]:
                plt.plot(X_D[j].cpu(), sample.cpu(), color=colors[task], alpha=0.1)

            plt.plot(X_D[j].cpu(), Y_D_pmeans[task][j].mean(0).cpu(), color=colors[task])
            plt.plot(X_D[j].cpu(), Y_D[task][j].cpu(), color='k')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        vis = PIL.Image.open(buf)
        vis = toten(vis)

        logger.writer.add_image('valid_samples_{}'.format(task), vis, global_step=logger.global_step)
        plt.close()
    logger.writer.flush()
    

class Logger():
    def __init__(self, log_dir, reset=True):
        if os.path.exists(log_dir) and reset:
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
        self.logs = {}
        self.logs_saved = {}
        self.iters = {}
        
    def register_key(self, key):
        self.logs[key] = 0
        self.logs_saved[key] = 0
        self.iters[key] = 0
        
    def add_value(self, key, value):
        self.logs[key] += value
        self.iters[key] += 1
        
    def get_value(self, key):
        if self.iters[key] == 0:
            return self.logs_saved[key]
        else:
            return self.logs[key] / self.iters[key]
        
    def reset(self, keys):
        for key in keys:
            self.logs_saved[key] = self.get_value(key)
            self.logs[key] = 0
            self.iters[key] = 0
            
    def log_values(self, keys, pbar=None, tag='train', global_step=0):
        if pbar is not None:
            desc = 'step {:05d}'.format(global_step)
            for key in keys:
                name = key.replace('nll_', '')
                name = name.replace('kld', 'kl')
                name = name.replace('sine', 'sn')
                name = name.replace('tanh', 'th')
                name = name.replace('sigmoid', 'sg')
                name = name.replace('gaussian', 'gs')
                desc += ', {}: {:.3f}'.format(name, self.get_value(key))
            pbar.set_description(desc)
        for key in keys:
            self.writer.add_scalar('{}/{}'.format(tag, key), self.get_value(key), global_step=global_step)