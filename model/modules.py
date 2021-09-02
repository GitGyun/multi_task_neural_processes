import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, SAB, PMA
from .utils import masked_forward


class FFB(nn.Module):
    def __init__(self, dim_in, dim_out, act_fn, ln, dr):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out) if ln else nn.Identity(),
            act_fn(),
            nn.Dropout(dr),
        )
    
    def forward(self, x):
        return self.layers(x)
        

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, act_fn=nn.ReLU, ln=False, dr=0., skip=False):
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
            layers.append(FFB(di, do, act_fn, ln, dr))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for l_idx in range(len(self.layers)):
            if self.skip:
                if ((l_idx == 0 and self.dim_in != self.dim_hidden) or \
                    (l_idx == len(self.layers) - 1 and self.dim_out != self.dim_hidden)):
                    x = self.layers[l_idx](x)
                else:
                    x = x + self.layers[l_idx](x)
            else:
                x = self.layers[l_idx](x)
        
        return x
        

class STEncoder(nn.Module):
    def __init__(self, dim_hidden, n_layers, n_heads, act_fn=nn.ReLU, ln=False, dr=0.1, pool=True):
        super().__init__()
        self.layers = nn.ModuleList([
            SAB(dim_hidden, n_heads, ln=ln, dr=dr)
            for _ in range(n_layers)
        ])
        if pool:
            self.pool = PMA(dim_hidden, n_heads, 1, act_fn=act_fn, ln=ln, dr=dr)
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
                 n_layers=2, act_fn=nn.ReLU, ln=False, dr=0., skip=False,
                 epsilon=0.1, sigma=True, sigma_act=torch.sigmoid):
        super().__init__()
        
        self.epsilon = epsilon
        self.sigma = sigma
        
        assert n_layers >= 2
        self.mlp = MLP(dim_in, dim_hidden, dim_hidden, n_layers-1, act_fn, ln, dr, skip)
        
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
    def __init__(self, dim_q, dim_k, dim_v, n_layers, n_heads, act_fn=nn.ReLU, ln=False, dr=0.1, proj=True):
        super().__init__()
        self.proj = proj
        if self.proj:
            self.query_proj = nn.Linear(dim_q, dim_v)
            self.key_proj = nn.Linear(dim_k, dim_v)
        self.attentions =  nn.ModuleList([Attention(dim_v, n_heads, act_fn=act_fn, ln=ln, dr=dr)
                                          for _ in range(n_layers)])
        
    def forward(self, Q, K, V, **kwargs):
        if self.proj:
            Q = self.query_proj(Q)
            K = self.key_proj(K)
        for attention in self.attentions:
            Q = attention(Q, K, V, **kwargs)
            
        return Q
    
    
class MultiTaskAttention(nn.Module):
    def __init__(self, dim_hidden, n_layers, n_heads, act_fn=nn.ReLU, ln=False, dr=0.1, pool=False):
        super().__init__()
        layers = [SAB(dim_hidden, n_heads, act_fn=act_fn, ln=ln, dr=dr) for _ in range(n_layers)]
        self.pool = pool
        if self.pool:
            layers.append(PMA(dim_hidden, n_heads, 1, act_fn=act_fn, ln=ln, dr=dr))
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
        
def conv_block(ic, oc, ks, st, pd, act_fn, dr=0):
    layers = [nn.Conv2d(ic, oc, ks, st, pd),
              nn.BatchNorm2d(oc),
              act_fn()]
    if dr > 0:
        layers.append(nn.Dropout(dr))
    return nn.Sequential(*layers)


def deconv_block(ic, oc, ks, st, pd, act_fn, dr=0):
    layers = [nn.ConvTranspose2d(ic, oc, ks, st, pd),
              nn.BatchNorm2d(oc),
              act_fn()]
    if dr > 0:
        layers.append(nn.Dropout(dr))
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_fn=nn.ReLU, dropout=False):
        super().__init__()
        self.dim_out = dim_out 
        dr = 0.5 if dropout else 0
        self.blocks = nn.Sequential(
            conv_block(dim_in, dim_hidden, 3, 1, 1, act_fn, dr),
            conv_block(dim_hidden, dim_hidden*2, 4, 2, 1, act_fn, dr),
            conv_block(dim_hidden*2, dim_hidden*4, 4, 2, 1, act_fn, dr),
            conv_block(dim_hidden*4, dim_hidden*8, 4, 2, 1, act_fn, dr),
            conv_block(dim_hidden*8, dim_hidden*8, 4, 2, 1, act_fn, dr)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Sequential(
            conv_block(dim_hidden*8, dim_out, 1, 1, 0, act_fn, dr),
            nn.Conv2d(dim_out, dim_out, 1)
        )

    def forward(self, x):
        hidden = self.blocks(x)
        hidden = self.avgpool(hidden)
        hidden = self.out(hidden).view(x.size(0), self.dim_out)
        return hidden


class ConvDecoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_fn=nn.ReLU, dropout=False):
        super().__init__()
        assert dim_hidden % 8 == 0
        dr = 0.5 if dropout else 0
        self.proj_head = nn.Conv2d(dim_in, dim_hidden, 1)
        self.blocks = nn.Sequential(
            deconv_block(dim_hidden, dim_hidden, 4, 2, 1, act_fn, dr),
            deconv_block(dim_hidden, dim_hidden//2, 4, 2, 1, act_fn, dr),
            deconv_block(dim_hidden//2, dim_hidden//4, 4, 2, 1, act_fn, dr),
            deconv_block(dim_hidden//4, dim_hidden//8, 4, 2, 1, act_fn, dr),
            deconv_block(dim_hidden//8, dim_hidden//8, 3, 1, 1, act_fn, dr),
        )
        self.out = nn.Sequential(
            conv_block(dim_hidden//8, dim_out, 1, 1, 0, act_fn, dr),
            nn.Conv2d(dim_out, dim_out, 1)
        )

    def forward(self, x):
        hidden = self.proj_head(x.unsqueeze(2).unsqueeze(3))
        hidden = self.blocks(hidden)
        hidden = self.out(hidden)
        return hidden