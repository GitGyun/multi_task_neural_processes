import os

import torch
from torch.utils.data import Dataset
from torch.distributions import Bernoulli

from .utils import mask_labels


activations = {
    'sine': lambda x: torch.sin(x),
    'tanh': lambda x: torch.tanh(x),
    'sigmoid': lambda x: torch.sigmoid(x),
    'gaussian': lambda x: torch.exp(-x.pow(2))
}


class SyntheticDataset(Dataset):
    def __init__(self, data_path, split, tasks, split_ratio, target_size):
        assert len(split_ratio) == 3
        self.tasks = tasks
        # parse data root
        if os.path.isdir(data_path):
            self.root = data_path
        else:
            self.root = os.path.split(data_path)[0]
            
        self.n_functions = 1000
        self.n_points = 200
        self.target_size = target_size
        assert self.target_size <= self.n_points or split == 'test'
        
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
        
        # load data
        X, Y, meta_info = torch.load(data_path)
        self.X = X[self.function_idxs]
        self.Y = {task: Y[task][self.function_idxs] for task in tasks}
        self.meta_info = meta_info
        
    def __len__(self):
        return len(self.function_idxs)
    
    
class SyntheticTrainDataset(SyntheticDataset):
    def __getitem__(self, idx):
        '''
        Returns complete target.
        '''
        # generate point permutation
        pp = torch.randperm(self.n_points)[:self.target_size]
        # pp = torch.randperm(self.n_points)
        
        # get permuted data
        X_D = self.X[idx][pp].clone()
        Y_D = {task: self.Y[task][idx][pp].clone() for task in self.tasks}
        
        return X_D, Y_D
    
    
class SyntheticTestDataset(SyntheticDataset):
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
            
        # generate target data on a uniform grid
        self.generate_target_data(target_size, tasks)
        
    def generate_target_data(self, target_size, tasks):
        self.X_D = torch.stack([torch.linspace(-5, 5, target_size).unsqueeze(1) for _ in self.function_idxs])
        self.Y_D = {
            task: torch.stack([
                self.meta_info[function_idx]['a'] * \
                activations[task](self.meta_info[function_idx]['w']*self.X_D[i] + self.meta_info[function_idx]['b']) + \
                self.meta_info[function_idx]['c']
                for i, function_idx in enumerate(self.function_idxs)
            ])
            for task in tasks
        }
        
    def __getitem__(self, idx):
        '''
        Returns incomplete context, complete target, and complete context labels.
        '''
        # pick context
        context_idxs = self.pp[idx][:self.context_size]
        
        X_C = self.X[idx][context_idxs].clone()
        Y_C = {task: self.Y[task][idx][context_idxs].clone() for task in self.tasks}
        
        # predict all points as target
        X_D = self.X_D[idx].clone()
        Y_D = {task: self.Y_D[task][idx].clone() for task in self.tasks}
        
        # random drop with mask
        Y_C_comp = {task: Y_C[task].clone() for task in Y_C} # copy unmasked context labels
        if self.gamma > 0:
            mask_labels(Y_C, self.mask[idx, :self.context_size], self.tasks)
            
        # gt parameters
        gt_params = self.meta_info[self.function_idxs[idx]]

        return X_C, Y_C, X_D, Y_D, Y_C_comp, gt_params
        
        
class SyntheticTNTestDataset(SyntheticTestDataset):
    def generate_target_data(self, target_size, tasks):
        self.X_D = torch.stack([torch.linspace(-5, 5, target_size).unsqueeze(1) for _ in self.function_idxs])
        self.Y_D = {
            task: torch.stack([
                self.meta_info[function_idx][task]['a'] * \
                activations[task](self.meta_info[function_idx][task]['w']*self.X_D[i] + \
                                  self.meta_info[function_idx][task]['b']) + \
                self.meta_info[function_idx][task]['c']
                for i, function_idx in enumerate(self.function_idxs)
            ])
            for task in tasks
        }