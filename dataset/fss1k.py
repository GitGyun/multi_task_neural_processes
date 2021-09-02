import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class FSSDataset(Dataset):
    def __init__(self, data_path, split, ways=5, shots=5, base_size=(64, 64)):
        self.data_path = data_path
        self.split = split
        self.ways = ways
        self.shots = shots
        self.base_size = base_size
        
        assert shots < 10
        assert 240 % ways == 0
        
        with open(os.path.join(self.data_path, 'test_classes.txt'), 'r') as f:
            all_classes = os.listdir(os.path.join(self.data_path, 'images'))
            test_classes = list(map(lambda x: x.strip('\n'), f.readlines()))
            if self.split == 'test':
                self.categories = test_classes
            else:
                trainval_classes = sorted(list(set(all_classes).difference(set(test_classes))))
                assert len(trainval_classes) == 760
                if self.split == 'train':
                    self.categories = trainval_classes[:520]
                else:
                    self.categories = trainval_classes[520:]
                    
        self.toten = ToTensor()
        
    def load_image(self, category, instance):
        img_path = os.path.join(self.data_path, 'images', category, f'{instance}.jpg')
        img = Image.open(img_path).resize(self.base_size)
        x = self.toten(img)
        
        return x
    
    def load_label(self, category, instance, label):
        mask_path = os.path.join(self.data_path, 'images', category, f'{instance}.png')
        mask = Image.open(mask_path).convert('L').resize(self.base_size)
        y = self.toten(mask).squeeze(0)
        y = (label + 1)*(y > 0.5).long()
        
        return y
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError
        

class FSSTrainDataset(FSSDataset):
    def __len__(self):
        return 3141592
    
    def __getitem__(self, idx):
        # sample classes
        classes = np.random.choice(self.categories, self.ways, replace=False)
        
        # load image and label
        X_C = []
        Y_C = []
        X_D = []
        Y_D = []
        for label, c in enumerate(classes):
            # sample instances
            all_instances = range(1, 11)
            contexts = np.random.choice(all_instances, self.shots, replace=False)
            targets = set(all_instances).difference(contexts)
            
            for i in contexts:
                X_C.append(self.load_image(c, i))
                Y_C.append(self.load_label(c, i, label))
            
            for i in targets:
                X_D.append(self.load_image(c, i))
                Y_D.append(self.load_label(c, i, label))
                
        X_C = torch.stack(X_C)
        Y_C = torch.stack(Y_C)
        X_D = torch.stack(X_D)
        Y_D = torch.stack(Y_D)
        
        return X_C, Y_C, X_D, Y_D

    
class FSSTestDataset(FSSDataset):
    def __len__(self):
        return 240 // self.ways
    
    def __getitem__(self, idx):
        # determine classes and instances
        classes = self.categories[idx*self.ways:(idx+1)*self.ways]
        contexts = range(1, 1+self.shots)
        targets = range(1+self.shots, 11)
        
        # load image and label
        X_C = []
        Y_C = []
        X_D = []
        Y_D = []
        for label, c in enumerate(classes):
            for i in contexts:
                X_C.append(self.load_image(c, i))
                Y_C.append(self.load_label(c, i, label))
            
            for i in targets:
                X_D.append(self.load_image(c, i))
                Y_D.append(self.load_label(c, i, label))
                
        X_C = torch.stack(X_C)
        Y_C = torch.stack(Y_C)
        X_D = torch.stack(X_D)
        Y_D = torch.stack(Y_D)
        
        return X_C, Y_C, X_D, Y_D