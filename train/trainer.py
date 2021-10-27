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
    
#     B, N = X_D.size()[:2]
#     loss = F.cross_entropy(Y_D_logits.view(B*N, *Y_D_logits.size()[2:]),
#                            Y_D.view(B*N, *Y_D.size()[2:]))

    Y_D_binary = torch.stack([(Y_D == idx_c + 1).float() for idx_c in range(config.ways)], 2)
    loss = F.binary_cross_entropy(torch.sigmoid(Y_D_logits), Y_D_binary.float())
        
    logger.add_value('cross_entropy', loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update global step
    logger.global_step += 1
    
    if logger.global_step % config.val_iter == 0:
        if config.ways == 1:
            Y_D_pred = (Y_D_logits.squeeze(2) > 0).long()
        else:
            Y_D_pred = torch.where(Y_D_logits.max(2)[0] > 0,
                                   1+torch.argmax(Y_D_logits, 2),
                                   torch.zeros_like(Y_D_logits[:, :, 0]).long())
        plot_images(X_D.cpu(), Y_D.cpu(), Y_D_pred.cpu(), config.ways, config.shots, logger, tag='train')
    

@torch.no_grad()
def inference(model, X_C, Y_C, X_D, binary=True):
    model.eval()
    Y_D_logits = model(X_C, Y_C, X_D)
    model.train()
    if binary:
        Y_D_pred = (Y_D_logits.squeeze(2) > 0).long()
    else:
        Y_D_pred = torch.where(Y_D_logits.max(2)[0] > 0,
                               1+torch.argmax(Y_D_logits, 2),
                               torch.zeros_like(Y_D_logits[:, :, 0]).long())
    return Y_D_pred



def evaluate(model, test_loader, device, config, logger=None, tag='valid', n_vis=8):
    '''
    Calculate error of model based on the posterior predictive mean.
    '''
    calculator = RunningConfusionMatrix(config.ways+1)
    
    n_datasets = 0
    n_vis = max(1, n_vis // config.ways)
    vis_dict = {'X_D': [], 'Y_D': [], 'Y_D_pred': [], 'n_vis': 0}
        
    for b_idx, test_data in enumerate(test_loader):
        X_C, Y_C, X_D, Y_D = to_device(test_data, device)

        Y_D_pred = inference(model, X_C, Y_C, X_D, binary=(config.ways == 1))
        calculator.update_matrix(Y_D.reshape(-1).cpu(), Y_D_pred.reshape(-1).cpu())
        
        # plot single batch
        if logger is not None:
            if vis_dict['n_vis'] < n_vis:
                vis_dict['X_D'].append(X_D[:min(X_D.size(0), n_vis - vis_dict['n_vis'])].cpu())
                vis_dict['Y_D'].append(Y_D[:min(X_D.size(0), n_vis - vis_dict['n_vis'])].cpu())
                vis_dict['Y_D_pred'].append(Y_D_pred[:min(X_D.size(0), n_vis - vis_dict['n_vis'])].cpu())
                vis_dict['n_vis'] += min(X_D.size(0), n_vis - vis_dict['n_vis'])
            elif vis_dict['n_vis'] == n_vis:
                plot_images(torch.cat(vis_dict['X_D']),
                            torch.cat(vis_dict['Y_D']),
                            torch.cat(vis_dict['Y_D_pred']), config.ways, config.shots, logger)
                vis_dict['n_vis'] += 1
            
    miou = calculator.compute_current_mean_intersection_over_union()

    if logger is not None:
        logger.writer.add_scalar(f'{tag}/miou', miou, global_step=logger.global_step)
        logger.writer.flush()
    
    return miou