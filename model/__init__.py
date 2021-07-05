from .models import STP, JTP, MTP

model_class = {'stp': STP, 'jtp': JTP, 'mtp': MTP}

def get_model(config, device):
    return model_class[config.model](config.dim_x, config.dim_ys, config.dim_hidden,
                                     config.tasks, config.layernorm, config.n_attn_heads,
                                     config.module_sizes).to(device)