from torch.utils.data import DataLoader

from .utils import get_data_iterator, to_device
from .fss1k import FSSTrainDataset, FSSTestDataset


def load_data(config, device, split='trainval'):
    '''
    Load train & valid or test data and return the iterator & loader.
    '''
    assert config.data == 'fss1k'
    TrainDataset, TestDataset = FSSTrainDataset, FSSTestDataset
        
    
    # load train iterator
    if split == 'train' or split == 'trainval':
        train_data = TrainDataset(config.data_path, 'train', config.ways, config.shots, config.base_size, config.crop_size)
        train_loader = DataLoader(train_data, batch_size=config.global_batch_size,
                                  shuffle=True, pin_memory=(device.type == 'cuda'),
                                  drop_last=True, num_workers=config.num_workers)
        train_iterator = get_data_iterator(train_loader, device)
        
    # load valid loader
    if split == 'valid' or split == 'trainval':
        valid_data = TestDataset(config.data_path, 'valid', config.ways, config.shots, config.base_size, config.crop_size)
        valid_loader = DataLoader(valid_data, batch_size=config.global_batch_size,
                                  shuffle=False, pin_memory=(device.type == 'cuda'),
                                  drop_last=False, num_workers=config.num_workers)
    
    # load test loader
    if split == 'test':
        test_data = TestDataset(config.data_path, 'test', config.ways, config.shots, config.base_size, config.crop_size)
        test_loader = DataLoader(test_data, batch_size=config.global_batch_size,
                                 shuffle=False, pin_memory=(device.type == 'cuda'),
                                 drop_last=False, num_workers=config.num_workers)

    # return
    if split == 'trainval':
        return train_iterator, valid_loader
    elif split == 'train':
        return train_iterator
    elif split == 'valid':
        return valid_loader
    elif split == 'test':
        return test_loader
    else:
        raise NotImplementedError