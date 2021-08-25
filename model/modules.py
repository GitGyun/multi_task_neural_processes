import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, SAB, PMA
from .utils import masked_forward


def MLP(dim_in, dim_out, dim_hidden, n_layers):
    layers = [nn.Linear(dim_in, dim_hidden)]
    for _ in range(n_layers-2):
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dim_hidden, dim_hidden))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)
        

class STEncoder(nn.Module):
    def __init__(self, dim_hidden, n_attn_heads, layernorm, n_layers, pool=True):
        super().__init__()
        self.layers = nn.ModuleList([
            SAB(dim_hidden, n_attn_heads, ln=layernorm)
            for _ in range(n_layers)
        ])
        if pool:
            self.pool = PMA(dim_hidden, n_attn_heads, 1, ln=layernorm)
        else:
            self.pool = None
            
    def forward(self, x, mask=None):
        for layer in self.layers:    
            x = layer(x, mask)
            
        if self.pool is not None:
            return self.pool(x).squeeze(1)
        else:
            return x


class LatentMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden,
                 n_layers=2, epsilon=0.1, sigma=True, sigma_act=torch.sigmoid):
        super().__init__()
        
        self.epsilon = epsilon
        self.sigma = sigma
        
        layers = [nn.Linear(dim_in, dim_hidden)]
        for _ in range(n_layers-2):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
        self.layers = nn.Sequential(*layers)
        
        self.hidden_to_mu = nn.Linear(dim_hidden, dim_out)
        if self.sigma:
            self.hidden_to_log_sigma = nn.Linear(dim_hidden, dim_out)
            self.sigma_act = sigma_act
        
    def forward(self, x):
        hidden = F.relu(self.layers(x))
        
        mu = self.hidden_to_mu(hidden)
        if self.sigma:
            log_sigma = self.hidden_to_log_sigma(hidden)
            sigma = self.epsilon + (1 - self.epsilon)*self.sigma_act(log_sigma)
            
            return mu, sigma
        else:
            return mu
        
        
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads, ln=False, num_layers=2):
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
    def __init__(self, dim_hidden, num_heads, ln=False, num_layers=1, pool=False):
        super().__init__()
        layers = [SAB(dim_hidden, num_heads, ln=ln) for _ in range(num_layers)]
        self.pool = pool
        if self.pool:
            layers.append(PMA(dim_hidden, num_heads, 1, ln=ln))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, Q):
        bs, nb, ts, _ = Q.size()
        Q_ = Q.transpose(1, 2).reshape(bs*ts, nb, -1)
        Q_ = self.layers(Q_) # (bs*ts, nb, dim_hidden) or (bs*ts, 1, dim_hidden)
        
        Q = Q_.reshape(bs, ts, *Q_.size()[1:]).transpose(1, 2)
        if self.pool:
            return Q.squeeze(1)
        else:
            return Q