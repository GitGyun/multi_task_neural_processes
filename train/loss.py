import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

# from .utils import RunningConfusionMatrix


def compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T, config, logger=None):
    '''
    Compute (prior-approximated) elbo objective for NP-based models.
    '''
    log_prob = 0
    for task in p_Y:
        log_prob_ = Normal(*p_Y[task]).log_prob(Y_D[task]).mean(0).sum()
        log_prob += log_prob_
        if logger is not None:
            logger.add_value(f'nll_{task}', -log_prob_.item())
    if logger is not None:
        logger.add_value('nll_normalized', -log_prob.item() / len(config.tasks) / Y_D[task].size(1))
    
    kld_G = 0
    if q_D_G is not None:
        kld_G = kl_divergence(Normal(*q_D_G), Normal(*q_C_G)).mean(0).sum()
        if logger is not None:
            logger.add_value('kld_G', kld_G.item())

    kld_T = 0
    if q_D_T is not None:
        for block in q_D_T:
            kld_T_ = kl_divergence(Normal(*q_D_T[block]), Normal(*q_C_T[block])).mean(0).sum()
            kld_T += kld_T_
            if logger is not None:
                logger.add_value(f'kld_{block}', kld_T_.item())
        if logger is not None:
            logger.add_value('kld_T_normalized', kld_T.item() / len(config.tasks) / Y_D[task].size(1))
        
    elbo = log_prob - (config.beta_G*kld_G + config.beta_T*kld_T)
    
    return elbo


def compute_normalized_nll(Y_D, p_Y):
    nll = {}
    for task in Y_D:
        nll[task] = -Normal(*p_Y[task]).log_prob(Y_D[task]).mean()
    return nll


def compute_error(Y_D, Y_D_pred, scales=None):
    '''
    Compute (normalized) MSE for continuous tasks and 1 - mIoU for categorical tasks.
    '''
    error = {}
    for task in Y_D:
        mse = (Y_D[task] - Y_D_pred[task]).pow(2)
        if scales is not None:
            if isinstance(scales, dict):
                mse /= scales[task].pow(2)
            else:
                mse /= scales.pow(2)
        error[task] = mse.sum(-1).mean().cpu()
            
    return error

# def compute_error(Y_D, Y_D_pred, scales=None):
#     '''
#     Compute (normalized) MSE for continuous tasks and 1 - mIoU for categorical tasks.
#     '''
#     error = {}
#     for task in Y_D:
#         if task == 'segment':
#             calculator = RunningConfusionMatrix(None)
#             calculator.update_matrix(torch.argmax(Y_D[task], -1).reshape(-1).cpu(), Y_D_pred[task].reshape(-1).cpu())
#             miou = calculator.compute_current_mean_intersection_over_union()
#             error[task] = 1 - miou
#         else:
#             mse = (Y_D[task] - Y_D_pred[task]).pow(2)
#             if scales is not None:
#                 if isinstance(scales, dict):
#                     mse /= scales[task].pow(2)
#                 else:
#                     mse /= scales.pow(2)
#             error[task] = mse.mean().cpu()
            
#     return error

