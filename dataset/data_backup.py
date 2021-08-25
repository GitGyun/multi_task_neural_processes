import os
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Bernoulli
from torchvision.transforms import ToTensor, ToPILImage


from .generate_data import activations




class SyntheticTrainDataset(Dataset):
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

        return self.X[idx, perm].clone(), {task: self.Y[task][idx, perm].clone() for task in self.Y}
    
    
class SyntheticTestDataset(Dataset):
    def __init__(self, data_path, datasets, tasks, N, M, gamma, seed, split='test', noised=False, tasknoised=False):
        '''
        Test dataset samples (X_D, Y_D).
        '''
        super().__init__()
        X, Y, meta_info = torch.load(data_path)
        
        # save scales
        if tasknoised:
            self.scales = [{task: meta_info[dataset][task]['a'] for task in tasks} for dataset in datasets]
        else:
            self.scales = [meta_info[dataset]['a'] for dataset in datasets]

        # prepare context data
        self.X_C = X[datasets, :M]
        self.Y_C = {task: Y[task][datasets, :M] for task in tasks}
        self.M = M
        
        # randomly drop context labels with pre-computed masks
        if gamma > 0:
            mask_path = os.path.join('data', 'mask_indices', 'mask_M{}_gamma{}_seed{}_{}'.format(M, gamma, seed, split))
            assert os.path.exists(mask_path)
            mask = torch.load(mask_path)
            for t, task in enumerate(tasks):
                self.Y_C[task] = torch.where(mask[:, :M, t].unsqueeze(-1).expand_as(self.Y_C[task]),
                                             float('nan')*self.Y_C[task],
                                             self.Y_C[task])
                
        # generate random noise
        if noised:
            noise_path = os.path.join('data', 'noises', 'noise_N{}_seed{}_{}'.format(N, seed, split))
            if not os.path.exists(noise_path):
                os.makedirs(os.path.join('data', 'noises'), exist_ok=True)
                self.noise = {
                    task: torch.stack([
                        0.05*meta_info[dataset]['a']*torch.randn(N, 1)
                        for dataset in datasets
                    ])
                    for task in tasks
                }
                torch.save(self.noise, noise_path)
            else:
                self.noise = torch.load(noise_path)
        self.noised = noised
        self.tasknoised = tasknoised
        
        # prepare target data
        self.X_D = torch.stack([torch.linspace(-5, 5, N).unsqueeze(1) for dataset in datasets])
        if tasknoised:
            self.Y_D = {
                task: torch.stack([
                    meta_info[dataset][task]['a'] * \
                    activations[task](meta_info[dataset][task]['w']*self.X_D[i] + meta_info[dataset][task]['b']) + \
                    meta_info[dataset][task]['c']
                    for i, dataset in enumerate(datasets)
                ])
                for task in tasks
            }
        else:
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
        if self.noised:
            self.noise = [{task: self.noise[task][i] for task in tasks} for i in range(len(datasets))]
        
    def __len__(self):
        return len(self.X_C)
    
    def __getitem__(self, idx):
        if self.noised:
            X_C = self.X_C[idx].clone()
            Y_C = {task: self.Y_C[idx][task].clone() + self.noise[idx][task][:self.M].clone() for task in self.Y_C[idx]}
            X_D = self.X_D[idx].clone()
            Y_D = {task: self.Y_D[idx][task].clone() + self.noise[idx][task].clone() for task in self.Y_D[idx]}
            Y_D_denoised = {task: self.Y_D[idx][task].clone() for task in self.Y_D[idx]}
            if self.tasknoised:
                scales = {task: self.scales[idx][task].clone() for task in self.scales[idx]}
            else:
                scales = self.scales[idx].clone()
            
            return X_C, Y_C, X_D, Y_D, Y_D_denoised, scales
        else:
            X_C = self.X_C[idx].clone()
            Y_C = {task: self.Y_C[idx][task].clone() for task in self.Y_C[idx]}
            X_D = self.X_D[idx].clone()
            Y_D = {task: self.Y_D[idx][task].clone() for task in self.Y_D[idx]}
            scales = self.scales[idx].clone()

            return X_C, Y_C, X_D, Y_D, scales
    
    
class CelebADataset(Dataset):
    def __init__(self, data_path, datasets, tasks, N=400):
        super().__init__()
        self.data_path = data_path
        self.datasets = datasets
        # drop corrupted data
        corrupted_images = [5380, 5125]
        for ci in corrupted_images:
            if ci in self.datasets:
                self.datasets.pop(ci)
                
        self.tasks = tasks
        self.N = N
        self.subroots = {
            'rgb': 'CelebA-HQ-img-32',
            'edge': 'CelebAMask-HQ-sobel-32',
            'pncc': 'CelebAMask-HQ-pncc-32',
            'segment': 'CelebAMask-HQ-mask-32',
            }
        self.extensions = {
            'rgb': 'jpg',
            'edge': 'png',
            'pncc': 'png',
            'segment': 'png',
            }

        self.toten = ToTensor()
        self.topil = ToPILImage()
        self.coords = torch.stack(torch.meshgrid(torch.arange(32), torch.arange(32)), -1).div(32).mul(2).add(-1).reshape(-1, 2)
        
    def __len__(self):
        return len(self.datasets)
    

class CelebATrainDataset(CelebADataset):
    def __getitem__(self, idx):
        # permute pixels
        perm = torch.randperm(len(self.coords))[:self.N]
        X = self.coords[perm].clone()
        Y = {}
        
        # load image or label map
        for task in self.tasks:
            Y[task] = self.toten(Image.open(os.path.join(self.data_path, self.subroots[task],
                                                         "{}.{}".format(str(self.datasets[idx]), self.extensions[task]))))
            Y[task] = Y[task].reshape(Y[task].shape[0], -1).t()[perm]
            if task == "segment":
                Y[task] = F.one_hot((Y[task] * 255).squeeze(-1).long(), 19).float()
        
        return X, Y
    

class CelebATestDataset(CelebADataset):
    def __init__(self, data_path, datasets, tasks, N, M, gamma, seed, split, **kwargs):
        super().__init__(data_path, datasets, tasks, N)
        self.M = M
        self.gamma = gamma
        
        # load permutations
        perm_path = os.path.join(data_path, 'permutations_{}.pth'.format(split))
        if os.path.exists(perm_path):
            self.permutations = torch.load(perm_path)
        else:
            self.permutations = torch.stack([torch.randperm(N) for _ in datasets])
            torch.save(self.permutations, perm_path)
            print('initialized and saved {} permutations'.format(split))
    
        # load masks
        if gamma > 0:
            self.masks = torch.load(os.path.join('data', 'mask_indices', 'mask_M{}_gamma{}_seed{}_{}_celeba'.format(M, gamma, seed, split)))
                
    def __getitem__(self, idx):
        # load all pixels and permute
        perm = self.permutations[idx]
        X_D = self.coords[perm][:self.N].clone()
        Y_D = {}
        
        # load image or label map
        for task in self.tasks:
            Y_D[task] = self.toten(Image.open(os.path.join(self.data_path, self.subroots[task],
                                                           "{}.{}".format(str(self.datasets[idx]), self.extensions[task]))))
            Y_D[task] = Y_D[task].reshape(Y_D[task].shape[0], -1).t()[perm][:self.N]
            if task == "segment":
                Y_D[task] = F.one_hot((Y_D[task] * 255).squeeze(-1).long(), 19).float()
        
        # sample context
        X_C = X_D[:self.M].clone()
        Y_C = {task: Y_D[task][:self.M].clone() for task in Y_D}
        
        # mask context
        if self.gamma > 0:
            for t, task in enumerate(Y_C):
                Y_C[task] = torch.where(self.masks[idx, :self.M, t].unsqueeze(-1).expand_as(Y_C[task]),
                                        float('nan')*Y_C[task],
                                        Y_C[task])
        
        return X_C, Y_C, X_D, Y_D
    
    



