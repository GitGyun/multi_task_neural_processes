import os
import torch

from .mtnp import MTP, STP, JTP, SharedMTP


def get_model(config, device):
    if config.model == 'mtp':
        return MTP(config).to(device)
    elif config.model == 'stp':
        return STP(config).to(device)
    elif config.model == 'jtp':
        return JTP(config).to(device)
    elif config.model == 'mtp_s':
        return SharedMTP(config).to(device)
    else:
        raise NotImplementedError
        
        
def get_imputer(config, device):
    if config.model == 'jtp' and config.gamma_valid > 0:
        assert os.path.exists(config.imputer_path)
        ckpt_imputer = torch.load(config.imputer_path)
        params_imputer = ckpt_imputer['model']
        config_imputer = ckpt_imputer['config']
        imputer = get_model(config_imputer, device)
        imputer.load_state_dict_(params_imputer)
    else:
        imputer = config_imputer = None
        
    return imputer, config_imputer