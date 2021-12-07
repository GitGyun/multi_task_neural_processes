import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


def compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T, config, logger=None):
    '''
    Compute (prior-approximated) elbo objective for NP-based models.
    '''
    log_prob = 0
    for task in p_Y:
        if config.task_types[task] == 'continuous':
            log_prob_ = Normal(p_Y[task][0], p_Y[task][1]).log_prob(Y_D[task]).mean(0).sum()
        else:
            log_prob_ = -F.cross_entropy(p_Y[task].transpose(1, 2), torch.argmax(Y_D[task], -1), reduction='none').mean(0).sum()
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
        for task in q_D_T:
            kld_T_ = kl_divergence(Normal(*q_D_T[task]), Normal(*q_C_T[task])).mean(0).sum()
            kld_T += kld_T_
            if logger is not None:
                logger.add_value(f'kld_{task}', kld_T_.item())
        if logger is not None:
            logger.add_value('kld_T_normalized', kld_T.item() / len(config.tasks) / Y_D[task].size(1))
        
    elbo = log_prob - (config.beta_G*kld_G + config.beta_T*kld_T)
    
    return elbo


def compute_normalized_nll(Y_D, p_Y, task_types):
    nll = {}
    for task in Y_D:
        if task_types[task] == 'continuous':
            nll[task] = -Normal(p_Y[task][0], p_Y[task][1]).log_prob(Y_D[task]).mean()
        else:
            nll[task] = F.cross_entropy(p_Y[task].transpose(1, 2), torch.argmax(Y_D[task], -1))
    return nll


def compute_error(Y_D, Y_D_pred, task_types, scales=None):
    '''
    Compute (normalized) MSE
    '''
    error = {}
    for task in Y_D:
        if isinstance(scales, dict):
            scale = scales[task].pow(2)
        elif scales is not None:
            scale = scales.pow(2)
        else:
            scale = torch.ones(Y_D_pred[task].size(0), 1, 1, device=Y_D_pred[task].device)

        mse = ((Y_D_pred[task] - Y_D[task]).pow(2) / scale).sum(-1).mean()
        error[task] = mse.cpu()
            
    return error
