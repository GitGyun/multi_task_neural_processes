import torch
import torch.nn.functional as F

from .utils import plot_images, RunningConfusionMatrix
from dataset import to_device


def train_step(model, optimizer, config, logger, *train_data):
    '''
    Perform a training step.
    '''
    # forward
    X_C, Y_C, X_D, Y_D = train_data
    Y_D_logits = model(X_C, Y_C, X_D)
    loss = F.cross_entropy(Y_D_logits, Y_D)
    logger.add_value('cross_entropy', loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update global step
    logger.global_step += 1
    

@torch.no_grad()
def inference(model, X_C, Y_C, X_D):
    model.eval()
    Y_D_logits = model(X_C, Y_C, X_D)
    model.train()
    return torch.argmax(Y_D_logits, 1)



def evaluate(model, test_loader, device, config, logger=None, tag='valid'):
    '''
    Calculate error of model based on the posterior predictive mean.
    '''
    calculator = RunningConfusionMatrix(config.ways+1)
    
    n_datasets = 0
    for b_idx, test_data in enumerate(test_loader):
        X_C, Y_C, X_D, Y_D = to_device(test_data, device)

        Y_D_pred = inference(model, X_C, Y_C, X_D)
        calculator.update_matrix(Y_D.reshape(-1).cpu(), Y_D_pred.reshape(-1).cpu())
        
        # plot single batch
        if logger is not None and b_idx == 0:
            plot_images(X_D.cpu(), Y_D.cpu(), Y_D_pred.cpu(), logger, config.ways)
            
    miou = calculator.compute_current_mean_intersection_over_union()

        
    if logger is not None:
        logger.writer.add_scalar(f'{tag}/miou', miou, global_step=logger.global_step)
    
    return miou