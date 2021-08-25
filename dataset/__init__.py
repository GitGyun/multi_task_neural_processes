from torch.utils.data import DataLoader

from .utils import get_data_iterator, to_device, TrainCollator
from .template import TemplateTrainDataset, TemplateTestDataset
from .synthetic import SyntheticTrainDataset, SyntheticTestDataset


def load_data(config, device, split='trainval'):
    '''
    Load train & valid or test data and return the iterator & loader.
    '''
    if config.data == 'template':
        TrainDataset, TestDataset = TemplateTrainDataset, TemplateTestDataset
    elif config.data == 'synthetic':
        TrainDataset, TestDataset = SyntheticTrainDataset, SyntheticTestDataset
    else:
        raise NotImplementedError
        
    
    # load train iterator
    if split == 'train' or split == 'trainval':
        train_data = TrainDataset(config.data_path, 'train', config.tasks, config.split_ratio)
        train_collator = TrainCollator(config.ts_train, config.cs_range_train, config.gamma_train, config.task_blocks)
        train_loader = DataLoader(train_data, batch_size=config.global_batch_size,
                                  shuffle=True, pin_memory=(device.type == 'cuda'),
                                  drop_last=True, num_workers=config.num_workers, collate_fn=train_collator)
        train_iterator = get_data_iterator(train_loader, device)
        
    # load valid loader
    if split == 'valid' or split == 'trainval':
        valid_data = TestDataset(config.data_path, 'valid', config.tasks, config.task_blocks, config.split_ratio,
                                 config.ts_valid, config.cs_valid, config.gamma_valid, config.seed)
        valid_loader = DataLoader(valid_data, batch_size=config.global_batch_size,
                                  shuffle=False, pin_memory=(device.type == 'cuda'),
                                  drop_last=False, num_workers=config.num_workers)
    
    # load test loader
    if split == 'test':
        test_data = TestDataset(config.data_path, 'test', config.tasks, config.task_blocks, config.split_ratio,
                                config.ts_test, config.cs_test, config.gamma_test, config.seed)
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