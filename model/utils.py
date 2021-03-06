import torch


def masked_forward(module, x, mask, out_dim, **kwargs):
    assert x.size()[:-1] == mask.size()
    out = torch.zeros(*mask.size(), out_dim).to(x.device)
    out[~mask] = module(x[~mask], **kwargs)
    
    return out