import torch
import torch.nn as nn
import math

from .utils import masked_forward


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, act_fn=nn.ReLU, ln=False, dr=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_split = dim // num_heads
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.fc_o = nn.Linear(dim, dim, bias=False)
        
        self.activation = act_fn()
        self.attn_dropout = nn.Dropout(dr)
        self.residual_dropout1 = nn.Dropout(dr)
        self.residual_dropout2 = nn.Dropout(dr)
        if ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

    def forward(self, Q, K, V=None, mask_Q=None, mask_K=None, get_attn=False):
        if V is None: V = K
        
        if mask_Q is not None:
            Q = Q.clone().masked_fill(mask_Q.unsqueeze(-1), 0)
        else:
            mask_Q = torch.zeros(*Q.size()[:2], device=Q.device)
        
        if mask_K is not None:
            K = K.clone().masked_fill(mask_K.unsqueeze(-1), 0)
            V = V.clone().masked_fill(mask_K.unsqueeze(-1), 0)
        else:
            mask_K = torch.zeros(*K.size()[:2], device=K.device)
        
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        mask = ~((1 - mask_Q.unsqueeze(-1).float()).bmm((1 - mask_K.unsqueeze(-1).float()).transpose(1, 2)).bool().repeat(self.num_heads, 1, 1))
        
        A = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim)
        A = A.masked_fill(mask, -1e38)
        A = torch.softmax(A, 2)
        A = A.masked_fill(mask, 0)
            
        A = self.attn_dropout(A)
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        
        O = Q + self.residual_dropout1(O)
        O = O if getattr(self, 'ln1', None) is None else masked_forward(self.ln1, O, mask_Q.bool(), self.dim)
        O = O + self.residual_dropout2(self.activation(masked_forward(self.fc_o, O, mask_Q.bool(), self.dim)))
        O = O if getattr(self, 'ln2', None) is None else masked_forward(self.ln2, O, mask_Q.bool(), self.dim)
        
        if get_attn:
            return O, A
        else:
            return O


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, act_fn='relu', ln=False, dr=0.1):
        super().__init__()
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.attn = Attention(dim, num_heads, act_fn=act_fn, ln=ln, dr=dr)

    def forward(self, X, mask=None):
        return self.attn(self.S.repeat(X.size(0), 1, 1), X, mask_K=mask)
            
        
class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)
    
        
class SelfAttention(nn.Module):
    def __init__(self, dim, n_layers, n_heads, act_fn='relu', ln=False, dr=0.1):
        super().__init__()
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU
        
        self.attentions = nn.ModuleList([Attention(dim, n_heads, act_fn=act_fn, ln=ln, dr=dr)
                                         for _ in range(n_layers)])
        
    def forward(self, Q, mask=None, **kwargs):
        for attention in self.attentions:
            Q = attention(Q, Q, Q, mask_Q=mask, mask_K=mask, **kwargs)
            
        return Q
        
        
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, n_layers, n_heads, act_fn='relu', ln=False, dr=0.1):
        super().__init__()
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU
        
        self.query_proj = nn.Linear(dim_q, dim_v)
        self.key_proj = nn.Linear(dim_k, dim_v)
        self.attentions = nn.ModuleList([Attention(dim_v, n_heads, act_fn=act_fn, ln=ln, dr=dr)
                                         for _ in range(n_layers)])
        
    def forward(self, Q, K, V, **kwargs):
        Q = self.query_proj(Q)
        K = self.key_proj(K)
        for attention in self.attentions:
            Q = attention(Q, K, V, **kwargs)
            
        return Q