from .models import STP, JTP, MTP

model_class = {'stp': STP, 'jtp': JTP, 'mtp': MTP}

def get_model(config, device):
    return model_class[config.model](config.dim_x, config.dim_ys, config.dim_hidden,
                                     config.tasks, config.layernorm, config.n_attn_heads, config.epsilon,
                                     tuple(config.module_sizes),
                                     config.stochastic_path, config.deterministic_path,
                                     config.implicit_global_latent, config.global_latent_only,
                                     config.deterministic_path2, config.context_posterior).to(device)