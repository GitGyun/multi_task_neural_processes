import os
import sys
import shutil
import random
import io
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
# from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from torchvision.transforms import ToTensor
# from torchvision.utils import make_grid


def configure_experiment(config, args):
    # update config with arguments
    config.model = args.model
    config.architecture = args.architecture
    config.seed = args.seed
    config.name_postfix = args.name_postfix
    
    # parse arguments
    if args.n_steps > 0: config.n_steps = args.n_steps
    if args.lr > 0: config.lr = args.lr
    if args.global_batch_size > 0: config.global_batch_size = args.global_batch_size
    if args.dim_hidden > 0: config.dim_hidden = args.dim_hidden
    if len(args.module_sizes) > 0: config.module_sizes = [int(size) for size in args.module_sizes]
    if args.n_attn_heads > 0: config.n_attn_heads = args.n_attn_heads
    if args.dropout >= 0: config.dropout = args.dropout
        
    if args.layernorm is not None: config.layernorm = args.layernorm
    if args.skip is not None: config.skip = args.skip
    if args.task_embedding is not None: config.task_embedding = args.task_embedding
        
    if args.lr_schedule != '': config.lr_schedule = args.lr_schedule
    if args.beta_T_schedule != '': config.beta_T_schedule = args.beta_T_schedule
    if args.beta_G_schedule != '': config.beta_G_schedule = args.beta_G_schedule
    if args.activation != '': config.activation = args.activation

        
    # configure lvm
    if config.model == 'mtp':
        config.task_latents = config.global_latent = True
    elif config.model == 'mtp_glo':
        config.global_latent = True
        config.task_latents = False
    elif config.model == 'stp' or config.model == 'jtp':
        config.global_latent = False
        config.task_latents = True
        
    # configure architecture
    if config.architecture == 'np':
        config.stochastic_path = True
        config.deterministic_path = False
        config.local_deterministic_path = False
    elif config.architecture == 'cnp':
        config.stochastic_path = False
        config.deterministic_path = True
        config.local_deterministic_path = False
    elif config.architecture == 'anp':
        config.stochastic_path = True
        config.deterministic_path = False
        config.local_deterministic_path = True
    elif config.architecture == 'acnp':
        config.stochastic_path = False
        config.deterministic_path = True
        config.local_deterministic_path = True
    elif config.architecture == 'dnp':
        config.stochastic_path = True
        config.deterministic_path = True
        config.local_deterministic_path = False
    
    # configure task blocks
    if config.task_blocks is None:
        config.task_blocks = [[task] for task in config.tasks]
        
    if config.model == 'jtp':
        config.task_blocks_model = [config.tasks]
    else:
        config.task_blocks_model = config.task_blocks
        
    # configure training missing rate
    if config.model == 'stp' or config.model == 'jtp':
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
    exp_name = config.model + config.name_postfix
    os.makedirs('experiments', exist_ok=True)
    os.makedirs(os.path.join('experiments', config.log_dir), exist_ok=True)
    os.makedirs(os.path.join('experiments', config.log_dir, exp_name), exist_ok=True)
    log_dir = os.path.join('experiments', config.log_dir, exp_name, 'logs')
    save_dir = os.path.join('experiments', config.log_dir, exp_name, 'checkpoints')
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
    logger = Logger(log_dir)
    log_keys = ['nll_normalized'] + [f'nll_{task}' for task in config.tasks]
    if config.stochastic_path and config.global_latent:
        log_keys.append('kld_G')
    if config.stochastic_path and config.task_latents:
        log_keys += ['kld_T_normalized'] + [f'kld_{",".join(task_block)}' for task_block in config.task_blocks_model]
    for log_key in log_keys:
        logger.register_key(log_key)
    
    return logger, save_dir, log_keys


def plot_curves(logger, task_blocks, X_C, Y_C, X_D, Y_D, Y_C_comp, Y_D_pred, Y_C_imp=None, pred_type='map', size=3, markersize=5, n_subplots=10, n_row=5, colors=None):
    toten = ToTensor()
    plt.rc('xtick', labelsize=3*size)
    plt.rc('ytick', labelsize=3*size)
    
    if colors is None:
        colors = {task: 'k' for task in Y_D}

    for task_block in task_blocks:
        plt.figure(figsize=(size*n_row*4/3, size*(n_subplots // n_row)))
        for idx_sub in range(n_subplots):
            plt.subplot(n_subplots // n_row, n_row, idx_sub+1)
            
            x = X_D[idx_sub].squeeze(-1).cpu()
            p = torch.argsort(x)

            # split label dimensions into 1d
            for task in task_block:
                # plot target
                y = Y_D[task][idx_sub].squeeze(-1).cpu()
                plt.plot(x[p], y[p], color='k', alpha=0.5)

                # plot context
                index = ~Y_C[task][idx_sub].isnan()
                x_c = X_C[idx_sub, index].squeeze(-1).cpu()
                y_c = Y_C_comp[task][idx_sub, index].squeeze(-1).cpu()
                plt.scatter(x_c, y_c, color=colors[task], s=markersize*size)
                
                # plot imputed value
                x_s = X_C[idx_sub, ~index].squeeze(-1).cpu()
                if Y_C_imp is not None:
                    y_i = Y_C_imp[task][idx_sub, ~index].squeeze(-1).cpu()
                    plt.scatter(x_s, y_i, color=colors[task], s=markersize*size, marker='^')
                    
                # plot source
                y_s = Y_C_comp[task][idx_sub, ~index].squeeze(-1).cpu()
                plt.scatter(x_s, y_s, color=colors[task], s=1.5*markersize*size, marker='x')

                # plot predictions (either MAP or predictive means)
                if pred_type == 'pmeans':
                    for sample in Y_D_pred[task][:, idx_sub]:
                        y_ps = sample.squeeze(-1).cpu()
                        plt.plot(x[p], y_ps[p], color=colors[task], alpha=0.1)
                    
                    y_pm = Y_D_pred[task][:, idx_sub].mean(0).squeeze(-1).cpu()    
                    error = (y_pm - y).pow(2).mean().item()
                    plt.plot(x[p], y_pm[p], color=colors[task], label=f'{error:.3f}')
                    
                elif pred_type == 'map':
                    mu = Y_D_pred[task][0][idx_sub].squeeze(-1).cpu()
                    sigma = Y_D_pred[task][1][idx_sub].squeeze(-1).cpu()
                    nll = -Normal(mu, sigma).log_prob(y).mean().cpu().item()
                    
                    plt.plot(x[p], mu[p], color=colors[task], label=f'{nll:.3f}')
                    plt.fill_between(x[p], mu[p] - sigma[p], mu[p] + sigma[p], color=colors[task], alpha=0.2)

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
        logger.writer.add_image(f'valid_samples_{pred_type}_{",".join(task_block)}', vis, global_step=logger.global_step)
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
        if isinstance(data, tuple):
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
            

# class RunningConfusionMatrix():
#     """Running Confusion Matrix class that enables computation of confusion matrix
#     on the go and has methods to compute such accuracy metrics as Mean Intersection over
#     Union MIOU.
    
#     Attributes
#     ----------
#     labels : list[int]
#         List that contains int values that represent classes.
#     overall_confusion_matrix : sklean.confusion_matrix object
#         Container of the sum of all confusion matrices. Used to compute MIOU at the end.
#     ignore_label : int
#         A label representing parts that should be ignored during
#         computation of metrics
        
#     """
    
#     def __init__(self, labels, ignore_label=255):
        
#         self.labels = labels
#         self.ignore_label = ignore_label
#         self.overall_confusion_matrix = None
        
#     def update_matrix(self, ground_truth, prediction):
#         """Updates overall confusion matrix statistics.
#         If you are working with 2D data, just .flatten() it before running this
#         function.
#         Parameters
#         ----------
#         groundtruth : array, shape = [n_samples]
#             An array with groundtruth values
#         prediction : array, shape = [n_samples]
#             An array with predictions
#         """
        
#         # Mask-out value is ignored by default in the sklearn
#         # read sources to see how that was handled
#         # But sometimes all the elements in the groundtruth can
#         # be equal to ignore value which will cause the crush
#         # of scikit_learn.confusion_matrix(), this is why we check it here
#         if (ground_truth == self.ignore_label).all():
            
#             return
        
#         current_confusion_matrix = confusion_matrix(y_true=ground_truth,
#                                                     y_pred=prediction,
#                                                     labels=self.labels)
        
#         if self.overall_confusion_matrix is not None:
            
#             self.overall_confusion_matrix += current_confusion_matrix
#         else:
            
#             self.overall_confusion_matrix = current_confusion_matrix
    
#     def compute_current_mean_intersection_over_union(self):
        
#         intersection = np.diag(self.overall_confusion_matrix)
#         ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
#         predicted_set = self.overall_confusion_matrix.sum(axis=0)
#         union =  ground_truth_set + predicted_set - intersection

#         intersection_over_union = intersection / union.astype(np.float32)
#         mean_intersection_over_union = np.mean(intersection_over_union)
        
#         return mean_intersection_over_union
    
    
# def uint82bin(n, count=8):
#     """returns the binary of integer n, count refers to amount of bits"""
#     return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

# def labelcolormap(N):
#     if N == 19: # CelebAMask-HQ
#         cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
#                      (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
#                      (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
#                      (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
#                      (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
#                      dtype=np.uint8)
#     elif N == 2: # CelebAMask-HQ
#         cmap = np.array([(0,  0,  0), (255, 255,  255)],
#                      dtype=np.uint8)
#     else:
#         cmap = np.zeros((N, 3), dtype=np.uint8)
#         for i in range(N):
#             r, g, b = 0, 0, 0
#             id = i
#             for j in range(7):
#                 str_id = uint82bin(id)
#                 r = r ^ (np.uint8(str_id[-1]) << (7-j))
#                 g = g ^ (np.uint8(str_id[-2]) << (7-j))
#                 b = b ^ (np.uint8(str_id[-3]) << (7-j))
#                 id = id >> 3
#             cmap[i, 0] = r
#             cmap[i, 1] = g
#             cmap[i, 2] = b
#     return cmap


# class Colorize(object):
#     def __init__(self, n=35, cmap=None):
#         if cmap is None:
#             self.cmap = labelcolormap(n)
#         else:
#             self.cmap = cmap
#         self.cmap = self.cmap[:n]

#     def preprocess(self, x):
#         if len(x.size()) > 3 and x.size(1) > 1:
#             # if x has a shape of [B, C, H, W],
#             # where B and C denote a batch size and the number of semantic classes,
#             # then translate it into a shape of [B, 1, H, W]
#             x = x.argmax(dim=1, keepdim=True).float()
#         assert (len(x.shape) == 4) and (x.size(1) == 1), 'x should have a shape of [B, 1, H, W]'
#         return x

#     def __call__(self, x):
# #         x = self.preprocess(x)
# #         if (x.dtype == torch.float) and (x.max() < 2):
# #             x = x.mul(255).long()
#         x = x.unsqueeze(1)
#         color_images = []
#         gray_image_shape = x.shape[1:]
#         for gray_image in x:
#             color_image = torch.ByteTensor(3, *gray_image_shape[1:]).fill_(0)
#             for label, cmap in enumerate(self.cmap):
#                 mask = (label == gray_image[0]).cpu()
#                 color_image[0][mask] = cmap[0]
#                 color_image[1][mask] = cmap[1]
#                 color_image[2][mask] = cmap[2]

#             color_images.append(color_image)
#         color_images = torch.stack(color_images)
#         return color_images
    
    
# def plot_images(X_C, Y_C, X_D, Y_D, Y_D_pred, logger, Y_C_imp=None):
#     B = X_D.size(0)
#     coord_C = (X_C*16 + 16).long()
#     coord_D = (X_D*16 + 16).long()
#     for task in Y_D:
#         if task == 'segment':
#             context = torch.zeros(B, 32, 32)
#             gt = torch.zeros(B, 32, 32)
#             pred = torch.zeros(B, 32, 32)
            
#             for b_idx in range(B):
#                 context[b_idx, coord_C[b_idx, :, 0], coord_C[b_idx, :, 1]] = torch.argmax(Y_C[task][b_idx], -1).cpu().float()
#                 gt[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = torch.argmax(Y_D[task][b_idx], -1).cpu().float()
#                 pred[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = Y_D_pred[task][b_idx].cpu().float()

#             context = torch.where(context.isnan(), torch.zeros_like(context), context)
#             context = logger.colorizer(context).float().div(255)
#             pred = logger.colorizer(pred).float().div(255)
#             gt = logger.colorizer(gt).float().div(255)
#         else:
#             if task == 'edge':
#                 context = torch.zeros(B, 32, 32, 1)
#                 gt = torch.zeros(B, 32, 32, 1)
#                 pred = torch.zeros(B, 32, 32, 1)
#             else:
#                 context = torch.zeros(B, 32, 32, 3)
#                 gt = torch.zeros(B, 32, 32, 3)
#                 pred = torch.zeros(B, 32, 32, 3)
            
#             for b_idx in range(B):
#                 context[b_idx, coord_C[b_idx, :, 0], coord_C[b_idx, :, 1]] = Y_C[task][b_idx].clamp(0, 1).cpu()
#                 gt[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = Y_D[task][b_idx].clamp(0, 1).cpu()
#                 pred[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = Y_D_pred[task][b_idx].clamp(0, 1).cpu()
            
#             context = torch.where(context.isnan(), torch.zeros_like(context), context)
            
#             context = context.permute(0, 3, 1, 2)
#             gt = gt.permute(0, 3, 1, 2)
#             pred = pred.permute(0, 3, 1, 2)
            
#             if task == 'edge':
#                 context = context.repeat(1, 3, 1, 1)
#                 gt = gt.repeat(1, 3, 1, 1)
#                 pred = pred.repeat(1, 3, 1, 1)
        
#         vis = torch.cat((context, gt, pred))
#         vis = F.interpolate(vis, (128, 128), mode='nearest')
#         vis = make_grid(vis, nrow=B, padding=0)
#         logger.writer.add_image('valid_samples_{}'.format(task), vis, global_step=logger.global_step)
        
#     return vis 
