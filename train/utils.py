import os
import sys
import shutil
import random
import io
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from data import colors


def configure_experiment(config, args):
    # update config with arguments
    config.seed = args.seed
    config.name_postfix = args.name_postfix
    if args.n_steps > 0:
        config.n_steps = args.n_steps
    if args.lr > 0:
        config.lr = args.lr
    if args.global_batch_size > 0:
        config.global_batch_size = args.global_batch_size
    if len(args.module_sizes) > 0:
        config.module_sizes = [int(size) for size in args.module_sizes]
    if args.no_stochastic_path:
        config.stochastic_path = False
    if args.no_deterministic_path:
        config.deterministic_path = False
    if args.implicit_global_latent:
        config.implicit_global_latent = True
    if args.global_latent_only:
        config.global_latent_only = True
    else:
        config.global_latent_only = False
    if args.cnp_det:
        config.stochastic_path = config.context_posterior = False
        config.deterministic_path2 = True
    elif args.cnp_stc:
        config.stochastic_path = config.context_posterior = True
        config.deterministic_path2 = False
    else:
        config.deterministic_path2 = config.context_posterior = False
    if args.data == 'synthetic_noised' or args.data == 'synthetic_tasknoised':
        config.noised = True
    else:
        config.noised = False
    if args.data == 'synthetic_tasknoised':
        config.tasknoised = True
    else:
        config.tasknoised = False
    
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
        config.log_dir += '_debugging'

    # set directories
    exp_name = config.model + config.name_postfix
    os.makedirs('experiments', exist_ok=True)
    os.makedirs(os.path.join('experiments', config.log_dir), exist_ok=True)
    os.makedirs(os.path.join('experiments', config.log_dir, exp_name), exist_ok=True)
    log_path = os.path.join('experiments', config.log_dir, exp_name, 'logs')
    save_path = os.path.join('experiments', config.log_dir, exp_name, 'checkpoints')
    if os.path.exists(save_path):
        if args.debug_mode:
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


def plot_curves(X_C, Y_C, X_D, Y_D, Y_D_pmeans, logger, Y_C_imp=None, Y_D_denoised=None):
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
            if Y_D_denoised is not None:
                plt.plot(X_D[j].cpu(), Y_D_denoised[task][j].cpu(), color='k')
                plt.scatter(X_D[j].cpu(), Y_D[task][j].cpu(), color='k', s=3, alpha=0.3)
            else:
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
        self.colorizer = Colorize(19)
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
            
            

class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)
        
        if self.overall_confusion_matrix is not None:
            
            self.overall_confusion_matrix += current_confusion_matrix
        else:
            
            self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_current_mean_intersection_over_union(self):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union
    
    
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                     dtype=np.uint8)
    elif N == 2: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (255, 255,  255)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35, cmap=None):
        if cmap is None:
            self.cmap = labelcolormap(n)
        else:
            self.cmap = cmap
        self.cmap = self.cmap[:n]

    def preprocess(self, x):
        if len(x.size()) > 3 and x.size(1) > 1:
            # if x has a shape of [B, C, H, W],
            # where B and C denote a batch size and the number of semantic classes,
            # then translate it into a shape of [B, 1, H, W]
            x = x.argmax(dim=1, keepdim=True).float()
        assert (len(x.shape) == 4) and (x.size(1) == 1), 'x should have a shape of [B, 1, H, W]'
        return x

    def __call__(self, x):
#         x = self.preprocess(x)
#         if (x.dtype == torch.float) and (x.max() < 2):
#             x = x.mul(255).long()
        x = x.unsqueeze(1)
        color_images = []
        gray_image_shape = x.shape[1:]
        for gray_image in x:
            color_image = torch.ByteTensor(3, *gray_image_shape[1:]).fill_(0)
            for label, cmap in enumerate(self.cmap):
                mask = (label == gray_image[0]).cpu()
                color_image[0][mask] = cmap[0]
                color_image[1][mask] = cmap[1]
                color_image[2][mask] = cmap[2]

            color_images.append(color_image)
        color_images = torch.stack(color_images)
        return color_images
    
    
def plot_images(X_C, Y_C, X_D, Y_D, Y_D_pred, logger, Y_C_imp=None):
    B = X_D.size(0)
    coord_C = (X_C*16 + 16).long()
    coord_D = (X_D*16 + 16).long()
    for task in Y_D:
        if task == 'segment':
            context = torch.zeros(B, 32, 32)
            gt = torch.zeros(B, 32, 32)
            pred = torch.zeros(B, 32, 32)
            
            for b_idx in range(B):
                context[b_idx, coord_C[b_idx, :, 0], coord_C[b_idx, :, 1]] = torch.argmax(Y_C[task][b_idx], -1).cpu().float()
                gt[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = torch.argmax(Y_D[task][b_idx], -1).cpu().float()
                pred[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = Y_D_pred[task][b_idx].cpu().float()

            context = torch.where(context.isnan(), torch.zeros_like(context), context)
            context = logger.colorizer(context).float().div(255)
            pred = logger.colorizer(pred).float().div(255)
            gt = logger.colorizer(gt).float().div(255)
        else:
            if task == 'edge':
                context = torch.zeros(B, 32, 32, 1)
                gt = torch.zeros(B, 32, 32, 1)
                pred = torch.zeros(B, 32, 32, 1)
            else:
                context = torch.zeros(B, 32, 32, 3)
                gt = torch.zeros(B, 32, 32, 3)
                pred = torch.zeros(B, 32, 32, 3)
            
            for b_idx in range(B):
                context[b_idx, coord_C[b_idx, :, 0], coord_C[b_idx, :, 1]] = Y_C[task][b_idx].clamp(0, 1).cpu()
                gt[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = Y_D[task][b_idx].clamp(0, 1).cpu()
                pred[b_idx, coord_D[b_idx, :, 0], coord_D[b_idx, :, 1]] = Y_D_pred[task][b_idx].clamp(0, 1).cpu()
            
            context = torch.where(context.isnan(), torch.zeros_like(context), context)
            
            context = context.permute(0, 3, 1, 2)
            gt = gt.permute(0, 3, 1, 2)
            pred = pred.permute(0, 3, 1, 2)
            
            if task == 'edge':
                context = context.repeat(1, 3, 1, 1)
                gt = gt.repeat(1, 3, 1, 1)
                pred = pred.repeat(1, 3, 1, 1)
        
        vis = torch.cat((context, gt, pred))
        vis = F.interpolate(vis, (128, 128), mode='nearest')
        vis = make_grid(vis, nrow=B, padding=0)
        logger.writer.add_image('valid_samples_{}'.format(task), vis, global_step=logger.global_step)
        
    return vis 