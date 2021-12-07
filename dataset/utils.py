import numpy as np

import torch
from torch.distributions import Bernoulli

    
class TrainCollator:
    def __init__(self, target_size, context_size_range, gamma, tasks):
        self.target_size = target_size
        self.context_size_range = context_size_range
        self.gamma = gamma
        self.tasks = tasks
        
    def __call__(self, batch):
        X_D, Y_D = zip(*batch)
        X_D = torch.stack(X_D)
        Y_D = {task: torch.stack([Y_D_i[task] for Y_D_i in Y_D]) for task in Y_D[0]}
        batch_size = len(X_D)
        
        # sample context size
        if self.context_size_range is None:
            context_size = len(Y_D) + np.random.choice(self.target_size//2 - len(Y_D))
        else:
            if self.context_size_range[0] == self.context_size_range[1]:
                context_size = self.context_size_range[0]
            else:
                context_size = self.context_size_range[0] + \
                np.random.choice(self.context_size_range[1] - self.context_size_range[0])
        
        # sample context
        context_idxs = torch.randperm(self.target_size)[:context_size]
        X_C = X_D[:, context_idxs].clone()
        Y_C = {task: Y_D[task][:, context_idxs].clone() for task in Y_D}
        
        # simuate incompleteness
        if self.gamma > 0:
            # generate mask
            mask = Bernoulli(torch.tensor(self.gamma)).sample((batch_size, context_size, len(self.tasks))).bool()
            mask[:, :len(self.tasks)] *= ~torch.eye(len(self.tasks)).bool() # guarantee at least one label per task
            
            # mask labels
            mask_labels(Y_C, mask, self.tasks)
            
        return X_C, Y_C, X_D, Y_D


def mask_labels(Y, mask, tasks):
    assert len(tasks) == mask.shape[-1]
    for t_idx, task in enumerate(tasks):
        # fill the masked region with -1 (int) or nan (float)
        assert task in Y
        if Y[task].dtype == torch.int64:
            masked_tensor = -torch.ones_like(Y[task])
        else:
            masked_tensor = float('nan')*Y[task]

        Y[task] = torch.where(mask[..., t_idx].unsqueeze(-1).expand_as(Y[task]),
                              masked_tensor,
                              Y[task])
    

def to_device(data, device):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
            
    return to_device_wrapper(data)


def get_data_iterator(data_loader, device):
    '''
    Iterator wrapper for dataloader
    '''
    def get_batch():
        while True:
            for batch in data_loader:
                yield to_device(batch, device)
    return get_batch()