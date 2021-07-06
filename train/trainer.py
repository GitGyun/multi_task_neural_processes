import torch

from .loss import compute_elbo, compute_error
from .utils import plot_curves, plot_images
from data import to_device


def train_step(model, optimizer, config, logger, *train_data):
    '''
    Perform a training step.
    '''
    # forward
    X_C, Y_C, X_D, Y_D = train_data
    p_Y, q_D_G, q_C_G, q_D, q_C = model(X_C, Y_C, X_D, Y_D)
    loss = -compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D, q_C, config, logger)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update global step
    logger.global_step += 1
    
    
@torch.no_grad()
def inference_pmean(model, *test_data, K=1, L=1, get_pmeans=False):
    '''
    Calculate posterior predictive mean (or mode for categorical) with K global latents and L per-task latents.
    '''
    X_C, Y_C, X_D = test_data
    
    model.eval()
    p_Ys = model(X_C, Y_C, X_D, K=K, L=L)
    model.train()
    
    Y_D_pmeans = {}
    for task in Y_C:
        if task == 'segment':
            Y_D_pmeans[task] = torch.stack([p_Y[task].softmax(-1) for p_Y in p_Ys], 1)
        else:
            Y_D_pmeans[task] = torch.stack([p_Y[task].mean for p_Y in p_Ys], 1)
        
    Y_D_pred = {}
    for task in Y_C:
        if task == 'segment':
            Y_D_pred[task] = torch.argmax(Y_D_pmeans[task].mean(1), -1)
        else:
            Y_D_pred[task] = Y_D_pmeans[task].mean(1)
    
    if get_pmeans:
        return Y_D_pred, Y_D_pmeans
    else:
        return Y_D_pred



def evaluate(model, test_loader, device, config, logger=None,
             imputer=None, config_imputer=None, tag='valid'):
    '''
    Calculate error of model based on the posterior predictive mean.
    '''
    errors = {task: 0 for task in config.tasks}
    n_datasets = 0
    for b_idx, test_data in enumerate(test_loader):
        if config.data == 'synthetic':
            X_C, Y_C, X_D, Y_D, scales = to_device(test_data, device)
        elif config.data == 'celeba':
            X_C, Y_C, X_D, Y_D = to_device(test_data, device)
            scales = None
        
        # impute if imputer is given
        if imputer is not None:
            Y_C_input = Y_C_imp = inference_pmean(imputer, X_C, Y_C, X_C, K=config_imputer.K, L=config_imputer.L)
        else:
            Y_C_input = Y_C
            Y_C_imp = None
        
        # posterior predictive inference
        Y_D_pred, Y_D_pmeans = inference_pmean(model, X_C, Y_C_input, X_D, K=config.K, L=config.L, get_pmeans=True)
        
        # plot single batch
        if logger is not None:
            if config.data == 'synthetic' and b_idx == 0:
                plot_curves(X_C, Y_C, X_D, Y_D, Y_D_pmeans, logger, Y_C_imp)
            elif config.data == 'celeba' and b_idx == 1:
                plot_images(X_C, Y_C, X_D, Y_D, Y_D_pred, logger, Y_C_imp)

        # compute error
        errors_ = compute_error(Y_D, Y_D_pred, scales)
        
        # batch denormalization
        for task in config.tasks:
            errors[task] += (errors_[task]*X_C.size(0))
        n_datasets += X_C.size(0)

    # batch renormalization
    for task in config.tasks:
        errors[task] /= n_datasets
        
    if logger is not None:
        for task in config.tasks:
            if config.data == 'celeba':
                if task == 'segment':
                    logger.writer.add_scalar('{}/1-miou_{}'.format(tag, task), errors[task].item(),
                                             global_step=logger.global_step)
                else:
                    logger.writer.add_scalar('{}/mse_{}'.format(tag, task), errors[task].item(),
                                             global_step=logger.global_step)
            else:
                logger.writer.add_scalar('{}/nmse_{}'.format(tag, task), errors[task].item(),
                                         global_step=logger.global_step)
        
    return errors