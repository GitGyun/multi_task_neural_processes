import torch
import torch.nn as nn


class FFB(nn.Module):
    def __init__(self, dim_in, dim_out, act_fn, ln):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out) if ln else nn.Identity(),
            act_fn(),
        )
    
    def forward(self, x):
        return self.layers(x)

    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, act_fn='relu', ln=False):
        super().__init__()
        assert n_layers >= 1
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU
        
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        layers = []
        for l_idx in range(n_layers):
            di = dim_in if l_idx == 0 else dim_hidden
            do = dim_out if l_idx == n_layers - 1 else dim_hidden
            layers.append(FFB(di, do, act_fn, ln))
            
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        
        return x


class LatentMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers=2, act_fn='relu', ln=False,
                 epsilon=0.1, sigma=True, sigma_act=torch.sigmoid):
        super().__init__()
        
        self.epsilon = epsilon
        self.sigma = sigma
        
        assert n_layers >= 1
        if n_layers >= 2:
            self.mlp = MLP(dim_in, dim_hidden, dim_hidden, n_layers-1, act_fn, ln)
        else:
            self.mlp = None
        
        self.hidden_to_mu = nn.Linear(dim_hidden, dim_out)
        if self.sigma:
            self.hidden_to_log_sigma = nn.Linear(dim_hidden, dim_out)
            self.sigma_act = sigma_act
        
    def forward(self, x):
        hidden = self.mlp(x) if self.mlp is not None else x
        
        mu = self.hidden_to_mu(hidden)
        if self.sigma:
            log_sigma = self.hidden_to_log_sigma(hidden)
            sigma = self.epsilon + (1 - self.epsilon)*self.sigma_act(log_sigma)
            
            return mu, sigma
        else:
            return mu