import torch

from .loss import compute_elbo, compute_error, compute_normalized_nll
from .utils import plot_curves, broadcast_squeeze, broadcast_index, broadcast_mean
from dataset import to_device


def train_step(model, optimizer, config, logger, *train_data):
    '''
    Perform a training step.
    '''
    # forward
    X_C, Y_C, X_D, Y_D = train_data
    p_Y, q_D_G, q_C_G, q_D_T, q_C_T = model(X_C, Y_C, X_D, Y_D)
    loss = -compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T, config, logger)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update global step
    logger.global_step += 1

    
@torch.no_grad()
def inference_map(model, *test_data):
    '''
    Calculate map estimation (or mode for categorical) with K global latents and L per-task latents.
    '''
    X_C, Y_C, X_D = test_data
    
    model.eval()
    p_Ys = model(X_C, Y_C, X_D, MAP=True)
    model.train()
    
    return broadcast_squeeze(p_Ys, 0)
    
    
@torch.no_grad()
def inference_pmean(model, *test_data, ns_G=1, ns_T=1, get_pmeans=False):
    '''
    Calculate posterior predictive mean (or mode for categorical) with K global latents and L per-task latents.
    '''
    X_C, Y_C, X_D = test_data
    
    model.eval()
    p_Ys = model(X_C, Y_C, X_D, MAP=False, ns_G=ns_G, ns_T=ns_T)
    model.train()
    
    Y_D_pmeans = broadcast_index(p_Ys, 0)
#     for task in Y_C:
#         Y_D_pmeans[task] = torch.stack([p_Y[task][0] for p_Y in p_Ys], 1)
        
    Y_D_pred = broadcast_mean(Y_D_pmeans, 0)
#     Y_D_pred = {}
#     for task in Y_C:
#         Y_D_pred[task] = Y_D_pmeans[task].mean(1)
    
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
    nlls = {task: 0 for task in config.tasks}
        
    n_datasets = 0
    for b_idx, test_data in enumerate(test_loader):
        if config.data == 'synthetic':
            X_C, Y_C, X_D, Y_D, Y_C_comp, scales = to_device(test_data, device)
        else:
            X_C, Y_C, X_D, Y_D, Y_C_comp = to_device(test_data, device)
            scales = None
        
        # impute if imputer is given
        if imputer is not None:
            Y_C_input = Y_C_imp = inference_pmean(imputer, X_C, Y_C, X_C, ns_G=config_imputer.ns_G, ns_T=config_imputer.ns_T)
        else:
            Y_C_input = Y_C
            Y_C_imp = None
        
        # MAP inference
        Y_D_pred_map = inference_map(model, X_C, Y_C_input, X_D)
        # plot single batch
        if logger is not None and b_idx == 0:
            plot_curves(logger, config.task_blocks, X_C, Y_C, X_D, Y_D, Y_C_comp, Y_D_pred_map, Y_C_imp,
                        pred_type='map', colors=config.colors)
            
        # posterior predictive inference
        Y_D_pred, Y_D_pmeans = inference_pmean(model, X_C, Y_C_input, X_D, ns_G=config.ns_G, ns_T=config.ns_T, get_pmeans=True)
        # plot single batch
        if logger is not None and b_idx == 0:
            plot_curves(logger, config.task_blocks, X_C, Y_C, X_D, Y_D, Y_C_comp, Y_D_pmeans, Y_C_imp,
                        pred_type='pmeans', colors=config.colors)

        # compute errors
        nlls_ = compute_normalized_nll(Y_D, Y_D_pred_map)
        errors_ = compute_error(Y_D, Y_D_pred, scales)
        
        # batch denormalization
        for task in config.tasks:
            nlls[task] += (nlls_[task]*X_C.size(0))
            errors[task] += (errors_[task]*X_C.size(0))
        n_datasets += X_C.size(0)

    # batch renormalization
    for task in config.tasks:
        nlls[task] /= n_datasets
        errors[task] /= n_datasets
        
    if logger is not None:
        for task in config.tasks:
            logger.writer.add_scalar(f'{tag}/nll_{task}', nlls[task].item(),
                                     global_step=logger.global_step)
            logger.writer.add_scalar(f'{tag}/error_{task}', errors[task].item(),
                                     global_step=logger.global_step)
    
    return nlls, errors