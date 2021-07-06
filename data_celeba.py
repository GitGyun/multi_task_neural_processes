from pathlib import Path
from PIL import Image
from torchvision import transforms

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet, Bernoulli
import numpy as np


task_names = ['rgb', 'edge', 'pncc', 'segment']

class CelebADataset(Dataset):
    def __init__(self, data_path, datasets, tasks):
        super().__init__()
        N = 30000 # the size of CelebA-HQ dataset
        self.root = Path(data_path)
        self.datasets = datasets
        # drop corrupted data
        corrupted_images = [5380, 5125]
        for ci in corrupted_images:
            if ci in self.datasets:
                self.datasets.pop(ci)
                
        self.tasks = tasks
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

        self.toten = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        self.coords = torch.stack(torch.meshgrid(torch.arange(32), torch.arange(32)), -1).div(32).mul(2).add(-1).reshape(-1, 2)
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        # permute pixels
        perm = torch.randperm(len(self.coords))
        X = self.coords[perm]
        Y = {}
        
        # load image or label map
        for task in self.tasks:
            path = self.root / self.subroots[task] / ('%s.%s' % (str(self.datasets[idx]), self.extensions[task]))
            Y[task] = self.toten(Image.open(path))
            Y[task] = Y[task].reshape(Y[task].shape[0], -1).t()[perm]
            if task == "segment":
                Y[task] = (Y[task] * 255).squeeze(-1).float()
        
        return X, Y


def get_data_iterator(loader):
    def get_batch():
        while True:
            for batch in loader:
#                 loader.dataset.n_contexts = np.random.randint(len(args.tasks), args.n_targets//2+1, size = (1,))[0]
                yield batch
    return get_batch()



def load_data(args):
    train_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts=args.n_contexts, n_targets=args.n_targets, droprate=args.train_droprate, tasks=args.target_tasks, split='train')
    train_loader = DataLoader(train_data, batch_size=args.global_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    train_iterator = get_data_iterator(train_loader, args)
    
    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 10, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_a = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 30, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_b = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 100, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_c = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 1024, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_d = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    return train_iterator, valid_loader_a, valid_loader_b, valid_loader_c, valid_loader_d


def load_eval_data(args):
    train_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts=args.n_contexts, n_targets=args.n_targets, droprate=args.train_droprate, tasks=args.target_tasks, split='train')
    train_loader = DataLoader(train_data, batch_size=args.global_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    train_iterator = get_data_iterator(train_loader, args)
    
    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 10, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_a = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 30, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_b = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 100, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_c = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    valid_data = CelebADataset('/data4/wonkonge/mtp/datasets/CelebAMask-HQ/', n_contexts = 1024, n_targets = 1024, droprate = args.test_droprate, tasks=args.target_tasks, split='val')
    valid_loader_d = DataLoader(valid_data, batch_size=args.global_batch_size//2, shuffle=False, pin_memory=True, drop_last=False)

    return train_iterator, valid_loader_a, valid_loader_b, valid_loader_c, valid_loader_d


def to_device(data, device):
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
