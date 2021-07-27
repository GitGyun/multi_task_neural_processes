import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from .utils import RunningConfusionMatrix


def compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D, q_C, config, logger=None):
    '''
    Compute (prior-approximated) elbo objective for NP-based models.
    '''
    log_prob = 0
    for task in config.tasks:
        if task == 'segment':
            log_prob_ = -F.cross_entropy(p_Y[task].transpose(1, 2), torch.argmax(Y_D[task], -1), reduction='none').mean(0).sum()
        else:
            log_prob_ = p_Y[task].log_prob(Y_D[task]).mean(0).sum()
        log_prob += log_prob_
        if logger is not None:
            logger.add_value('nll_{}'.format(task), -log_prob_.item())
    
    kld_G = 0
    if q_D_G is not None:
        kld_G = kl_divergence(q_D_G, q_C_G).mean(0).sum()
        if logger is not None:
            logger.add_value('kld_G', kld_G.item())

    kld_task = 0
    if q_D is not None:
        for task in config.tasks:
            kld_task_ = kl_divergence(q_D[task], q_C[task]).mean(0).sum()
            kld_task += kld_task_
            if logger is not None:
                logger.add_value('kld_{}'.format(task), kld_task_.item())
        
    elbo = log_prob - config.beta*(kld_G + kld_task)
    
    return elbo


def compute_error(Y_D, Y_D_pred, scales=None):
    '''
    Compute (normalized) MSE for continuous tasks and 1 - mIoU for categorical tasks.
    '''
    error = {}
    for task in Y_D:
        if task == 'segment':
            calculator = RunningConfusionMatrix(None)
            calculator.update_matrix(torch.argmax(Y_D[task], -1).reshape(-1).cpu(), Y_D_pred[task].reshape(-1).cpu())
            miou = calculator.compute_current_mean_intersection_over_union()
            error[task] = 1 - miou
        else:
            mse = (Y_D[task] - Y_D_pred[task]).pow(2)
            if scales is not None:
                if isinstance(scales, dict):
                    mse /= scales[task].pow(2)
                else:
                    mse /= scales.pow(2)
            error[task] = mse.mean().cpu()
            
    return error

