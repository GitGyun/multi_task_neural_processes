import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .attentions import Attention, SAB, PMA, masked_forward

    
class STEncoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_h, n_heads=4, n_layers=3, ln=False, pool=False):
        super().__init__()
        self.dim_h = dim_h
        self.proj = nn.Linear(dim_x + dim_y, dim_h)
        self.layers = nn.Sequential(
            *[SAB(dim_h, n_heads, ln=ln) for _ in range(n_layers)]
        )
        if pool:
            self.pma = PMA(dim_h, 4, 1, ln=ln)
        self.pool = pool
        
    def forward(self, D, mask=None):
        if mask is None:
            s = self.proj(D)
            if self.pool:
                return self.pma(self.layers(s)).squeeze(1)
            else:
                return self.layers(s)
        else:
            s = masked_forward(self.proj, D, mask, (*D.size()[:-1], self.dim_h))
            for layer in self.layers:
                s = layer(s, mask=mask)
            if self.pool:
                return self.pma(s, mask=mask).squeeze(1)
            else:
                return s
            
            
class NormalEncoder(nn.Module):
    def __init__(self, dim_h, dim_z, epsilon=0.1):
        super().__init__()
        
        self.epsilon = epsilon
        self.h_to_hidden = nn.Linear(dim_h, dim_z)
        self.hidden_to_mu = nn.Linear(dim_z, dim_z)
        self.hidden_to_sigma = nn.Linear(dim_z, dim_z)
        
    def forward(self, h):
        hidden = F.relu(self.h_to_hidden(h)) # B x r_dim
        
        mu = self.hidden_to_mu(hidden) # B x z_dim
        sigma = self.epsilon + (1 - self.epsilon)*self.hidden_to_sigma(hidden).sigmoid() # B x z_dim
        q = Normal(mu, sigma)
        return q
    
    
class SingleTaskAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads, num_layers=2, ln=False):
        super().__init__()
        self.query_proj = nn.Linear(dim_q, dim_v)
        self.key_proj = nn.Linear(dim_k, dim_v)
        self.attentions =  nn.ModuleList([Attention(dim_v, num_heads, ln=ln) for _ in range(num_layers)])
        
    def forward(self, Q, K, V, **kwargs):
        Q = self.query_proj(Q)
        K = self.key_proj(K)
        for attention in self.attentions:
            Q = attention(Q, K, V, **kwargs)
            
        return Q
    
    
class MultiTaskAttention(nn.Module):
    def __init__(self, tasks, dim_q, dim_k, dim_v, num_heads, num_layers=3, ln=False):
        super().__init__()
        self.query_proj = nn.Linear(dim_q, dim_v)
        self.key_proj = nn.Linear(dim_k, dim_v)
        self.instance_attentions =  nn.ModuleList([
            nn.ModuleList([Attention(dim_v, num_heads, ln=ln) for _ in range(num_layers)])
            for task in tasks
        ])
        self.task_attentions = nn.ModuleList([SAB(dim_v, num_heads, ln=ln) for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, Q, K, V, masks=None):
        Q = self.query_proj(Q).unsqueeze(2).repeat(1, 1, len(V), 1)
        K = self.key_proj(K)
        if masks is None:
            masks = [None for _ in range(len(V))]
        for l_idx in range(self.num_layers):
            Q_ = []
            for t in range(len(V)):
                Q_.append(self.instance_attentions[t][l_idx](Q[:, :, t, :], K, V[t], mask_K=masks[t]).reshape(Q.size(0)*Q.size(1), -1))
            Q_ = torch.stack(Q_, 1)
            Q = self.task_attentions[l_idx](Q_).reshape(Q.size())
        
        return Q
            
            
class NormalDecoder(nn.Module):
    def __init__(self, dim_x, dim_hidden, dim_y, n_layers,
                 stochastic_path=True, deterministic_path=True):
        super().__init__()
        assert n_layers >= 1
        self.stochastic_path = stochastic_path
        self.deterministic_path = deterministic_path
        
        dim_input = dim_hidden
        if self.stochastic_path:
            dim_input += dim_hidden
        if self.deterministic_path:
            dim_input += dim_hidden
        
        self.input_proj = nn.Linear(dim_x, dim_hidden)
        
        layers = [nn.Linear(dim_input, dim_hidden)]
        for _ in range(n_layers-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.layers = nn.Sequential(*layers)
        
        self.hidden_to_mu = nn.Linear(dim_hidden, dim_y)
        self.hidden_to_sigma = nn.Linear(dim_hidden, dim_y)
        
    def forward(self, x, v=None, r=None):
        # prepare inputs
        x = self.input_proj(x)
        input_list = [x]
        if self.stochastic_path:
            assert v is not None
            input_list.append(v.unsqueeze(1).repeat(1, x.size(1), 1))
        if self.deterministic_path:
            assert r is not None
            input_list.append(r)
        
        # compute hidden state for normal parameters
        input_tensor = torch.cat(input_list, -1).reshape(x.size(0)*x.size(1), -1)
        hidden = self.layers(input_tensor).reshape(x.size(0), x.size(1), -1)

        # extract normal parameters from hidden state
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9*F.softplus(self.hidden_to_sigma(hidden))
        p = Normal(mu, sigma)
        
        return p