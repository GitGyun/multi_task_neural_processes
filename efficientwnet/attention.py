import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def masked_forward(module, x, mask, out_dim, **kwargs):
    if mask is None:
        out = module(x, **kwargs)
    else:
        assert x.size()[:-1] == mask.size()
        out = torch.zeros(*mask.size(), out_dim).to(x.device)
        out[~mask] = module(x[~mask], **kwargs)

    return out


class Attention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads=4, act_fn=nn.GELU, ln=True, dr=0.1, ff=True):
        super().__init__()
        self.dim = dim_V
        self.num_heads = num_heads
        self.dim_split = dim_V // num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=False)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=False)
        self.fc_v = nn.Linear(dim_V, dim_V, bias=False)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=False)
        
        self.activation = act_fn()
        self.attn_dropout = nn.Dropout(dr)
        
        self.ff = ff
        if self.ff:
            self.residual_dropout1 = nn.Dropout(dr)
            self.residual_dropout2 = nn.Dropout(dr)
            self.ln = ln
            if self.ln:
                self.ln1 = nn.LayerNorm(dim_V)
                self.ln2 = nn.LayerNorm(dim_V)
            
    def mask_tensors(self, Q, K, V, mask_Q, mask_K):
        masked = False
        if mask_Q is not None:
            Q = Q.clone().masked_fill(mask_Q.unsqueeze(-1), 0)
            masked = True
        else:
            mask_Q = torch.zeros(*Q.size()[:2], device=Q.device)
        
        if mask_K is not None:
            K = K.clone().masked_fill(mask_K.unsqueeze(-1), 0)
            V = V.clone().masked_fill(mask_K.unsqueeze(-1), 0)
            masked = True
        else:
            mask_K = torch.zeros(*K.size()[:2], device=K.device)
        
        if masked:
            mask = ~((1 - mask_Q.unsqueeze(-1).float()).bmm((1 - mask_K.unsqueeze(-1).float()).transpose(1, 2)).bool().repeat(self.num_heads, 1, 1))
        else:
            mask = None
            
        return Q, K, V, mask

    def forward(self, Q, K, V=None, mask_Q=None, mask_K=None, get_attn=False):
        if V is None: V = K
        
        # mask tensors
        Q, K, V, mask = self.mask_tensors(Q, K, V, mask_Q, mask_K)
        if mask_Q is not None:
            mask_Q = mask_Q.bool()
        
        # project tensors
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # split heads
        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)
        
        # (masked) multi-head attention
        A = Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim)
        if mask is not None:
            A = A.masked_fill(mask, -1e38)
        A = torch.softmax(A, 2)
        if mask is not None:
            A = A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        
        # residual feed-forward layers
        if self.ff:
            O = Q + self.residual_dropout1(O)
            O = masked_forward(self.ln1, O, mask_Q, self.dim) if self.ln else O
            O = O + self.residual_dropout2(self.activation(masked_forward(self.fc_o, O, mask_Q, self.dim)))
            O = masked_forward(self.ln2, O, mask_Q, self.dim) if self.ln else O

        if get_attn:
            return O, A
        else:
            return O

        
class SAB(nn.Module):
    def __init__(self, dim, num_heads=4, act_fn=nn.GELU, ln=True, dr=0.1, *args, **kwargs):
        super().__init__()
        self.attn = Attention(dim, dim, dim, num_heads, act_fn=act_fn, ln=ln, dr=dr)

    def forward(self, X, mask=None, **kwargs):
        return self.attn(X, X, mask_Q=mask, mask_K=mask, **kwargs)

        
class CAB(nn.Module):
    def __init__(self, dim, num_heads=4, act_fn=nn.GELU, ln=True, dr=0.1, *args, **kwargs):
        super().__init__()
        self.attn = Attention(dim, dim, dim, num_heads, act_fn=act_fn, ln=ln, dr=dr)

    def forward(self, Q, K, V=None, **kwargs):
        return self.attn(Q, K, V, **kwargs)
    

class PMA(nn.Module):
    def __init__(self, dim, num_heads=4, num_seeds=1, act_fn=nn.GELU, ln=True, dr=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.attn = Attention(dim, dim, dim, num_heads, act_fn=act_fn, ln=ln, dr=dr)

    def forward(self, X, mask=None):
        return self.attn(self.S.repeat(X.size(0), 1, 1), X, mask_K=mask)