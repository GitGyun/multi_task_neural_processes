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
from torch.distributions import Normal
from torchvision.transforms import ToTensor

from .schedulers import LRScheduler, HPScheduler


def configure_experiment(config, args):
    # update config with arguments
    config.model = args.model
    config.seed = args.seed
    config.name_postfix = args.name_postfix
    config.pma = args.pma
    
    # parse arguments
    if args.n_steps > 0: config.n_steps = args.n_steps
    if args.lr > 0: config.lr = args.lr
    if args.global_batch_size > 0: config.global_batch_size = args.global_batch_size
    if args.dim_hidden > 0: config.dim_hidden = args.dim_hidden
        
    if args.lr_schedule != '': config.lr_schedule = args.lr_schedule
    if args.beta_T_schedule != '': config.beta_T_schedule = args.beta_T_schedule
    if args.beta_G_schedule != '': config.beta_G_schedule = args.beta_G_schedule
    if args.gamma_train >= 0: config.gamma_train = args.gamma_train
    if args.gamma_valid >= 0: config.gamma_valid = args.gamma_valid
    if len(args.cs_range_train) > 0:
        assert len(args.cs_range_train) == 2
        config.cs_range_train = (int(args.cs_range_train[0]), int(args.cs_range_train[1]))
    if args.ts_train > 0:
        config.ts_train = args.ts_train
        
    # configure training missing rate
    if config.model == 'jtp':
        config.gamma_train = 0.
    
    # set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # for debugging
    if args.debug_mode:
        config.n_steps = 3
        config.log_iter = 1
        config.val_iter = 1
        config.save_iter = 1
        config.imputer_path = config.imputer_path.replace(config.log_dir, config.log_dir + '_debugging')
        config.log_dir += '_debugging'

    # set directories
    if args.log_root != '':
        config.log_root = args.log_root
    if args.name != '':
        exp_name = args.name
    else:
        exp_name = config.model + config.name_postfix
    if args.imputer_path != '':
        config.imputer_path = args.imputer_path
        
    os.makedirs(config.log_root, exist_ok=True)
    os.makedirs(os.path.join(config.log_root, config.log_dir), exist_ok=True)
    os.makedirs(os.path.join(config.log_root, config.log_dir, exp_name), exist_ok=True)
    log_dir = os.path.join(config.log_root, config.log_dir, exp_name, 'logs')
    save_dir = os.path.join(config.log_root, config.log_dir, exp_name, 'checkpoints')
    if os.path.exists(save_dir):
        if args.debug_mode:
            shutil.rmtree(save_dir)
        else:
            while True:
                print('redundant experiment name! remove existing checkpoints? (y/n)')
                inp = input()
                if inp == 'y':
                    shutil.rmtree(save_dir)
                    break
                elif inp == 'n':
                    print('quit')
                    sys.exit()
                else:
                    print('invalid input')
    os.makedirs(save_dir)

    # tensorboard logger
    logger = Logger(log_dir, config.tasks)
    log_keys = ['nll_normalized'] + [f'nll_{task}' for task in config.tasks]
    if config.model in ['mtp', 'mtp_s', 'jtp', 'mtp_s']:
        log_keys.append('kld_G')
    if config.model in ['mtp', 'stp', 'mtp_s']:
        log_keys += ['kld_T_normalized'] + [f'kld_{task}' for task in config.tasks]
    for log_key in log_keys:
        logger.register_key(log_key)
    
    return logger, save_dir, log_keys


def get_schedulers(optimizer, config):
    lr_scheduler = LRScheduler(optimizer, config.lr_schedule, config.lr, config.n_steps, config.lr_warmup)
    beta_G_scheduler = beta_T_scheduler = None
    if config.model in ['mtp', 'jtp', 'mtp_s']:
        beta_G_scheduler = HPScheduler(config, 'beta_G', config.beta_G_schedule, config.beta_G, config.n_steps, config.beta_G_warmup)
    if config.model in ['mtp', 'stp', 'mtp_s']:
        beta_T_scheduler = HPScheduler(config, 'beta_T', config.beta_T_schedule, config.beta_G, config.n_steps, config.beta_T_warmup)
    
    return lr_scheduler, beta_G_scheduler, beta_T_scheduler


def plot_curves(logger, tasks, X_C, Y_C, X_D, Y_D, Y_C_comp, Y_D_pred, Y_C_imp=None, pred_type='map', size=3, markersize=5, n_subplots=10, n_row=5, colors=None):
    toten = ToTensor()
    plt.rc('xtick', labelsize=3*size)
    plt.rc('ytick', labelsize=3*size)
    
    if colors is None:
        colors = {task: 'k' for task in Y_D}
        
    n_row = min(n_row, n_subplots)
    n_subplots = (n_subplots // n_row) * n_row

    for task in tasks:
        plt.figure(figsize=(size*n_row*4/3, size*(n_subplots // n_row)))
        for idx_sub in range(n_subplots):
            plt.subplot(n_subplots // n_row, n_row, idx_sub+1)

            index_D = ~Y_D[task][idx_sub].isnan()
            x_d = X_D[idx_sub, index_D].cpu()
            if len(x_d.size()) > 1:
                x_d = x_d.squeeze(-1)
            p_D = torch.argsort(x_d)

            # plot target
            y_d = Y_D[task][idx_sub, index_D].cpu()
            if len(y_d.size()) > 1:
                y_d = y_d.squeeze(-1)
            plt.plot(x_d[p_D], y_d[p_D], color='k', alpha=0.5)


            # pick observable context indices and sort them
            index_C = ~Y_C[task][idx_sub].isnan()
            x_c = X_C[idx_sub, index_C].cpu()
            if len(x_c.size()) > 1:
                x_c = x_c.squeeze(-1)
            p_C = torch.argsort(x_c)

            # plot context
            if Y_C_comp is not None:
                y_c = Y_C_comp[task][idx_sub, index_C].cpu()
            else:
                y_c = Y_C[task][idx_sub, index_C].cpu()
            if len(y_c.size()) > 1:
                y_c = y_c.squeeze(-1)
            plt.scatter(x_c[p_C], y_c[p_C], color=colors[task], s=markersize*size)


            # plot imputed value
            x_s = X_C[idx_sub, ~index_C].squeeze(-1).cpu()
            if Y_C_imp is not None:
                y_i = Y_C_imp[task][idx_sub, ~index_C].squeeze(-1).cpu()
                plt.scatter(x_s, y_i, color=colors[task], s=markersize*size, marker='^')

            # plot source
            if Y_C_comp is not None:
                y_s = Y_C_comp[task][idx_sub, ~index_C].squeeze(-1).cpu()
                plt.scatter(x_s, y_s, color=colors[task], s=1.5*markersize*size, marker='x')


            # plot predictions (either MAP or predictive means)
            if pred_type == 'pmeans':
                samples = Y_D_pred[task][idx_sub, :, index_D].cpu()
                for y_ps in samples:
                    plt.plot(x_d[p_D], y_ps[p_D], color=colors[task], alpha=0.1)

                y_pm = samples.mean(0)   
                error = (y_pm - y_d).pow(2).mean().item()
                plt.plot(x_d[p_D], y_pm[p_D], color=colors[task], label=f'{error:.3f}')

            elif pred_type == 'map':
                mu = Y_D_pred[task][0][idx_sub, index_D].cpu()
                sigma = Y_D_pred[task][1][idx_sub, index_D].cpu()
                nll = -Normal(mu, sigma).log_prob(y_d).mean().cpu().item()

                plt.plot(x_d[p_D], mu[p_D], color=colors[task], label=f'{nll:.3f}')
                plt.fill_between(x_d[p_D], mu[p_D] - sigma[p_D], mu[p_D] + sigma[p_D], color=colors[task], alpha=0.2)

            plt.legend()
            
        plt.tight_layout()

        # plt figure to io buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        # io buffer to tensor
        vis = PIL.Image.open(buf)
        vis = toten(vis)
        
        # log tensor and close figure
        logger.writer.add_image(f'valid_samples_{pred_type}_{task}', vis, global_step=logger.global_step)
        plt.close()
        
    logger.writer.flush()
    

class Logger():
    def __init__(self, log_dir, tasks, reset=True):
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
            
            if 'nll_normalized' in keys:
                desc += ', {}: {:.3f}'.format('nll_norm', self.get_value('nll_normalized'))
            if 'kld_T_normalized' in keys:
                desc += ', {}: {:.3f}'.format('kld_T_norm', self.get_value('kld_T_normalized'))
            if 'kld_G' in keys:
                desc += ', {}: {:.3f}'.format('kld_G', self.get_value('kld_G'))
            pbar.set_description(desc)

        for key in filter(lambda x: x not in ['nll_normalized', 'kld_T_normalized'], keys):
            self.writer.add_scalar('{}/{}'.format(tag, key), self.get_value(key), global_step=global_step)
            
        for key in filter(lambda x: x in ['nll_normalized', 'kld_T_normalized', 'kld_G'], keys):
            self.writer.add_scalar('{}_summary/{}'.format(tag, key), self.get_value(key), global_step=global_step)
            

class Saver:
    def __init__(self, model, save_dir, config):
        self.save_dir = save_dir
        self.config = config
        self.tasks = config.tasks
        self.model_type = config.model
        if self.model_type == 'stp':
            self.best_nll_state_dict = model.state_dict_()
            self.best_error_state_dict = model.state_dict_()
        
        self.best_nll = float('inf')
        self.best_nlls = {task: float('inf') for task in config.tasks}
        self.best_error = float('inf')
        self.best_errors = {task: float('inf') for task in config.tasks}
        
    def save(self, model, valid_nlls, valid_errors, global_step, save_name):
        torch.save({'model': model.state_dict_(), 'config': self.config,
                    'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                    os.path.join(self.save_dir, save_name))
        
    
    def save_best(self, model, valid_nlls, valid_errors, global_step):
        valid_nll = sum([valid_nlls[task] for task in self.tasks])
        valid_error = sum([valid_errors[task] for task in self.tasks])
        
        # save best model
        if self.model_type == 'stp':
            update_nll = False
            update_error = False
            for task in self.best_nlls:
                if valid_nlls[task] < self.best_nlls[task]:
                    self.best_nlls[task] = valid_nlls[task]
                    self.best_nll_state_dict[task] = model.state_dict_task(task)
                    update_nll = True
                    
                if valid_errors[task] < self.best_errors[task]:
                    self.best_errors[task] = valid_errors[task]
                    self.best_error_state_dict[task] = model.state_dict_task(task)
                    update_error = True
                    
            if update_nll:
                torch.save({'model': self.best_nll_state_dict, 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_nll.pth'))
            if update_error:
                torch.save({'model': self.best_error_state_dict, 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_error.pth'))
        else:
            if valid_nll < self.best_nll:
                self.best_nll = valid_nll
                torch.save({'model': model.state_dict_(), 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_nll.pth'))
            if valid_error < self.best_error:
                self.best_error = valid_error
                torch.save({'model': model.state_dict_(), 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_error.pth'))
    


def broadcast_squeeze(data, dim):
    def squeeze_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.squeeze(dim)
        elif isinstance(data, tuple):
            return tuple(map(squeeze_wrapper, data))
        elif isinstance(data, list):
            return list(map(squeeze_wrapper, data))
        elif isinstance(data, dict):
            return {key: squeeze_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
    
    return squeeze_wrapper(data)


def broadcast_index(data, idx):
    def index_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, tuple):
            return data[idx]
        elif isinstance(data, list):
            return data[idx]
        elif isinstance(data, dict):
            return {key: index_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
    
    return index_wrapper(data)


def broadcast_mean(data, dim):
    def mean_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.mean(dim)
        elif isinstance(data, tuple):
            return tuple(map(mean_wrapper, data))
        elif isinstance(data, list):
            return list(map(mean_wrapper, data))
        elif isinstance(data, dict):
            return {key: mean_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
    
    return mean_wrapper(data)