import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, SAB, PMA
from .utils import masked_forward


class FFB(nn.Module):
    def __init__(self, dim_in, dim_out, act_fn, dr, layernorm):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out) if layernorm else nn.Identity(),
            act_fn(),
            nn.Dropout(dr),
        )
    
    def forward(self, x):
        return self.layers(x)
        

class MLP(nn.Module):
    def __init__(dim_in, dim_out, dim_hidden, n_layers, act_fn=nn.ReLU, dr=0., layernorm=False, skip=False):
        super().__init__()
        assert n_layers >= 1
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.skip = skip

        layers = []
        for l_idx in range(n_layers):
            di = dim_in if l_idx == 0 else dim_hidden
            do = dim_out if l_idx == n_layers - 1 else dim_hidden
            layers.append(FFB(di, do, act_fn, dr, layernorm))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for l_idx in range(len(self.layers)):
            if self.skip:
                if ((l_idx == 0 and self.dim_in != self.dim_hidden) or \
                    (l_idx == len(self.layers) - 1 and self.dim_out != self.dim_hidden)):
                    x = self.blocks[l_idx](x)
                else:
                    x = x + self.blocks[l_idx](x)
            else:
                x = self.blocks[l_idx](x)
        
        return x
        

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
                 n_layers=2, act_fn=nn.ReLU, dr=0., layernorm=False, skip=False,
                 epsilon=0.1, sigma=True, sigma_act=torch.sigmoid):
        super().__init__()
        
        self.epsilon = epsilon
        self.sigma = sigma
        
        assert n_layers >= 2
        self.mlp = MLP(dim_in, dim_out, dim_hidden, n_layers-1, act_fn, dr, layernorm, skip)
        
        self.hidden_to_mu = nn.Linear(dim_hidden, dim_out)
        if self.sigma:
            self.hidden_to_log_sigma = nn.Linear(dim_hidden, dim_out)
            self.sigma_act = sigma_act
        
    def forward(self, x):
        hidden = self.mlp(x)
        
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