import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Bernoulli
import numpy as np

from .generate_data import activations


class TrainDataset(Dataset):
    def __init__(self, data_path, datasets, tasks):
        '''
        Train dataset samples (X_D, Y_D).
        '''
        super().__init__()
        X, Y, _ = torch.load(data_path)
        self.X = X[datasets]
        self.Y = {task: Y[task][datasets] for task in tasks}
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        perm = torch.randperm(self.X.size(1))

        return self.X[idx, perm], {task: self.Y[task][idx, perm] for task in self.Y}
    
    
class TestDataset(Dataset):
    def __init__(self, data_path, datasets, tasks, N, M, gamma, seed, split='test'):
        '''
        Test dataset samples (X_D, Y_D).
        '''
        super().__init__()
        X, Y, meta_info = torch.load(data_path)
        
        # save scales
        self.scales = [meta_info[dataset]['a'] for dataset in datasets]

        # prepare context data
        self.X_C = X[datasets, :M]
        self.Y_C = {task: Y[task][datasets, :M] for task in tasks}
        
        # randomly drop context labels with pre-computed masks
        if gamma > 0:
            mask = torch.load('data/mask_indices/mask_M{}_gamma{}_seed{}_{}'.format(M, gamma, seed, split))
            for t, task in enumerate(tasks):
                self.Y_C[task] = torch.where(mask[:, :M, t].unsqueeze(-1).expand_as(self.Y_C[task]),
                                             float('nan')*self.Y_C[task],
                                             self.Y_C[task])
        
        # prepare target data
        self.X_D = torch.stack([torch.linspace(-5, 5, N).unsqueeze(1) for dataset in datasets])
        self.Y_D = {
            task: torch.stack([
                meta_info[dataset]['a'] * \
                activations[task](meta_info[dataset]['w']*self.X_D[i] + meta_info[dataset]['b']) + \
                meta_info[dataset]['c']
                for i, dataset in enumerate(datasets)
            ])
            for task in tasks
        }
                
        # rearrange data
        self.Y_C = [{task: self.Y_C[task][i] for task in tasks} for i in range(len(datasets))]
        self.Y_D = [{task: self.Y_D[task][i] for task in tasks} for i in range(len(datasets))]
        
    def __len__(self):
        return len(self.X_C)
    
    def __getitem__(self, idx):
        return self.X_C[idx], self.Y_C[idx], self.X_D[idx], self.Y_D[idx], self.scales[idx]
    
    
def sample_context(X_D, Y_D, gamma=0):
    '''
    Sample (X_C, Y_C) from (X_D, Y_D) with random number of contexts 4 <= M <= N,
    then randomly drop task labels with missing rate gamma.
    '''
    B, N = X_D.size()[:2]
    M = 4 + np.random.choice(N//2 - 4)
    
    X_C = X_D[:, :M].clone()
    Y_C = {task: Y_D[task][:, :M].clone() for task in Y_D}
    
    if gamma > 0:
        # random dropping mask
        mask = Bernoulli(torch.tensor(gamma)).sample((B, M, len(Y_C))).bool()
        
        # Gaurantee that at least one label is contained for each task.
        mask[:, :len(Y_C)] *= (~torch.eye(len(Y_C), len(Y_C)).bool())
        
        # mask out context labels
        for t, task in enumerate(Y_C):
            Y_C[task] = torch.where(mask[..., t].unsqueeze(-1).expand_as(Y_C[task]),
                                    float('nan')*Y_C[task],
                                    Y_C[task])
    
    return X_C, Y_C


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


def get_train_iterator(train_loader, device, gamma=0):
    '''
    Iterator wrapper for train dataloader
    '''
    def get_batch():
        while True:
            for X_D, Y_D in train_loader:
                X_C, Y_C = sample_context(X_D, Y_D, gamma)
                batch = (X_C, Y_C, X_D, Y_D)
                yield to_device(batch, device)
    return get_batch()


def load_data(config, device, split='trainval'):
    '''
    Load train & valid or test data and return the iterator & loader.
    '''
    # load train iterator
    if split == 'train' or split == 'trainval':
        train_datasets = list(range(900))

        train_data = TrainDataset(config.data_path, train_datasets, config.tasks)
        train_loader = DataLoader(train_data, batch_size=config.global_batch_size,
                                  shuffle=True, pin_memory=(device.type == 'cuda'), drop_last=True, num_workers=4)
        train_iterator = get_train_iterator(train_loader, device, config.gamma_train)
        
    # load valid loader
    if split == 'valid' or split == 'trainval':
        valid_datasets = list(range(900, 950))
        valid_data = TestDataset(config.data_path, valid_datasets, config.tasks,
                                 config.N, config.M, config.gamma_test, config.seed, split='valid')
        valid_loader = DataLoader(valid_data, batch_size=config.global_batch_size,
                                  shuffle=False, pin_memory=(device.type == 'cuda'), drop_last=False, num_workers=4)
    
    # load test loader
    if split == 'test':
        test_datasets = list(range(950, 1000))

        test_data = TestDataset(config.data_path, test_datasets, config.tasks,
                                config.N, config.M, config.gamma_test, config.seed, split='test')
        test_loader = DataLoader(test_data, batch_size=config.global_batch_size,
                                 shuffle=False, pin_memory=(device.type == 'cuda'), drop_last=False, num_workers=4)

    # return
    if split == 'trainval':
        return train_iterator, valid_loader
    elif split == 'train':
        return train_iterator
    elif split == 'valid':
        return valid_loader
    elif split == 'test':
        return test_loader