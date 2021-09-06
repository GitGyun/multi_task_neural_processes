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


colors = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 1., 0.), (1., 0., 1.)]


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
        
    if args.lr_schedule != '': config.lr_schedule = args.lr_schedule
    if args.activation != '': config.activation = args.activation
        
    # configure model and architecture
    config.global_path = (config.architecture == 'acnp' or config.architecture == 'cnp')
    config.local_path = (config.architecture == 'acnp' or config.architecture == 'anp')
    config.inter_channel_attention = (config.model == 'mtp')
    
    config.base_size = (int(config.base_size[0]), int(config.base_size[1]))
    
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
    exp_name = config.model + '_' + config.architecture + config.name_postfix
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
    log_keys = ['cross_entropy']
    logger = Logger(log_dir)
    logger.register_key('cross_entropy')
    
    return logger, save_dir, log_keys
    

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
            desc = f'step {global_step:05d}, ce: {self.get_value("cross_entropy"):.3f}'
            pbar.set_description(desc)

        for key in keys:
            self.writer.add_scalar(f'{tag}/{key}'.format(tag, key), self.get_value(key), global_step=global_step)
            

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
        
        self.labels = list(range(labels))
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
    
    
def overlap_label(X, Y, n_classes=5, alpha=0.5):
    assert len(X.size()) == len(Y.size()) + 1
    if len(X.size()) == 3:
        return overlap_label(X.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)
    
    for c in range(n_classes):
        masked_region = (Y == (c + 1)).unsqueeze(1).expand_as(X)
        colored_mask = torch.tensor(colors[c]).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(X)
        overlapped_img = (1 - alpha)*X + alpha*colored_mask
        X = torch.where(masked_region, overlapped_img, X)
        
    return X

def color_label(Y, n_classes=5):
    if len(Y.size()) == 2:
        return color_label(Y.unsqueeze(0)).squeeze(0)
    
    label_map = torch.zeros_like(Y).float().unsqueeze(1).repeat(1, 3, 1, 1)
    for c in range(n_classes):
        masked_region = (Y == (c + 1)).unsqueeze(1).expand_as(label_map)
        colored_mask = torch.tensor(colors[c]).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(label_map)
        label_map = torch.where(masked_region, colored_mask, label_map)
    
    return label_map

def make_triple(X, Y, n_classes=5):
    assert len(X.size()) == 4
    batch_size = X.size(0)
    
    overlapped = overlap_label(X, Y)
    label_map = color_label(Y)
    vis = torch.cat((overlapped.reshape(n_classes, batch_size//n_classes, *overlapped.size()[1:]),
                     X.reshape(n_classes, batch_size//n_classes, *X.size()[1:]),
                     label_map.reshape(n_classes, batch_size//n_classes, *label_map.size()[1:])), 1)
    vis = vis.reshape(batch_size*3, *vis.size()[2:])
    vis = make_grid(vis, nrow=n_classes*3)
    
    return vis
    
    
def plot_images(X, Y, Y_pred, logger, n_classes):
    assert len(X.size()) == 5
    vis_gt = []
    vis_pred = []
    for b_idx in range(X.size(0)):
        vis_gt.append(make_triple(X[b_idx], Y[b_idx], n_classes))
        vis_pred.append(make_triple(X[b_idx], Y_pred[b_idx], n_classes))
    vis_gt = make_grid(torch.stack(vis_gt), nrow=1)
    vis_pred = make_grid(torch.stack(vis_pred), nrow=1)
    
    logger.writer.add_image('valid_gt', vis_gt, global_step=logger.global_step)
    logger.writer.add_image('valid_pred', vis_pred, global_step=logger.global_step)
