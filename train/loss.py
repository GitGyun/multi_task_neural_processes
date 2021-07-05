from torch.distributions import kl_divergence


def elbo(Y_D, p_Y, q_D_G, q_C_G, q_D, q_C, config, logger=None):
    '''
    Compute (prior-approximated) elbo objective for NP-based models.
    '''
    log_prob = 0
    for task in config.tasks:
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


def normalized_mse(Y_D, Y_D_pred, scales=None):
    '''
    Compute MSE normalized by the given scale.
    '''
    mse = {}
    for task in Y_D:
        mse_ = (Y_D[task] - Y_D_pred[task]).pow(2)
        if scales is not None:
            mse_ /= scales.pow(2)
        mse[task] = mse_.mean()
        
    return mse
    