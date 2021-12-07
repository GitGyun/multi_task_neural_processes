import os

import torch
from torch.utils.data import Dataset
from torch.distributions import Bernoulli

from .utils import mask_labels


class WeatherDataset(Dataset):
    def __init__(self, data_path, split, tasks, split_ratio, target_size):
        '''
        Train dataset samples (X_D, Y_D).
        '''
        super().__init__()
        
        self.tasks = tasks
        
        # parse data root
        if os.path.isdir(data_path):
            self.root = data_path
        else:
            self.root = os.path.split(data_path)[0]
            
        data = torch.load(data_path)
        
        # extract number of datasets and input points (timestamps).
        keys = sorted(list(data.keys()))
        self.n_functions = len(keys)
        self.n_points = data[keys[0]][tasks[0]].shape[1]
        self.target_size = target_size
        assert self.target_size <= self.n_points
        self.split = split
        
        # split function indices
        function_idxs = list(range(self.n_functions))
        if type(split_ratio[0]) is float:
            cut1 = int(len(function_idxs)*split_ratio[0])
            cut2 = cut1 + int(len(function_idxs)*split_ratio[1])
            cut3 = cut2 + int(len(function_idxs)*split_ratio[2])
        else:
            cut1 = split_ratio[0]
            cut2 = cut1 + split_ratio[1]
            cut3 = cut2 + split_ratio[2]
            
        # split functions
        if split == 'train':
            self.function_idxs = function_idxs[:cut1]
        elif split == 'valid':
            self.function_idxs = function_idxs[cut1:cut2]
        elif split == 'test':
            self.function_idxs = function_idxs[cut2:cut3]
        else:
            raise NotImplementedError
        
        # construct input and output tensors.
        self.X = torch.linspace(0, 1, self.n_points).unsqueeze(1)
        self.Y = {}
        for task in tasks:
            self.Y[task] = torch.stack([torch.from_numpy(data[keys[idx]][task]).transpose(0, 1).float()
                                        for idx in self.function_idxs])
        
    def __len__(self):
        return len(self.function_idxs)
    
    
class WeatherTrainDataset(WeatherDataset):
    def __getitem__(self, idx):
        '''
        Returns complete target.
        '''
        # generate point permutation
        pp = torch.randperm(self.n_points)[:self.target_size]
        # pp = torch.randperm(self.n_points)
        
        # get permuted data
        X_D = self.X[pp].clone()
        Y_D = {task: self.Y[task][idx][pp].clone() for task in self.tasks}
        
        return X_D, Y_D
    
    
class WeatherTestDataset(WeatherDataset):
    def __init__(self, data_path, split, tasks, split_ratio, target_size,
                 context_size, gamma, seed):
        '''
        Load or generate random objects.
        '''
        super().__init__(data_path, split, tasks, split_ratio, target_size)
        self.tasks = tasks
        self.context_size = context_size

        # generate or load point permutations of size (n_functions, n_points)
        os.makedirs(os.path.join(self.root, 'point_permutations'), exist_ok=True)
        pp_path = os.path.join(self.root, 'point_permutations', 'pp_seed{}_{}.pth'.format(seed, split))
        if os.path.exists(pp_path):
            self.pp = torch.load(pp_path)
        else:
            self.pp = [torch.randperm(self.n_points) for _ in range(len(self.function_idxs))]
            torch.save(self.pp, pp_path)
        
        # generate label mask of size (n_functions, n_points, n_tasks) with missing rate gamma
        self.gamma = gamma
        if self.gamma > 0:
            os.makedirs(os.path.join(self.root, 'label_masks'), exist_ok=True)
            mask_path = os.path.join(self.root, 'label_masks',
                                     f'mask_T{len(self.tasks)}_gamma{gamma}_seed{seed}_{split}.pth')
            if os.path.exists(mask_path):
                self.mask = torch.load(mask_path)
            else:
                self.mask = Bernoulli(torch.tensor(gamma)).sample((len(self.function_idxs),
                                                                   self.n_points, len(self.tasks))).bool()
                self.mask[:, :len(self.tasks)] *= ~torch.eye(len(self.tasks)).bool() # guarantee at least one label per task
                torch.save(self.mask, mask_path)
        else:
            self.mask = None
        
    def __getitem__(self, idx):
        '''
        Returns incomplete context, complete target, and complete context labels.
        '''
        # pick context
        context_idxs = self.pp[idx][:self.context_size]
        
        X_C = self.X[context_idxs].clone()
        Y_C = {task: self.Y[task][idx][context_idxs].clone() for task in self.tasks}
        
        # predict all points as target
        X_D = self.X.clone()
        Y_D = {task: self.Y[task][idx].clone() for task in self.tasks}
        
        # random drop with mask
        Y_C_comp = {task: Y_C[task].clone() for task in Y_C} # copy unmasked context labels
        if self.gamma > 0:
            mask_labels(Y_C, self.mask[idx, :self.context_size], self.tasks)

        return X_C, Y_C, X_D, Y_D, Y_C_comp
        
