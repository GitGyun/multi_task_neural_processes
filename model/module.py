import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP, LatentMLP
from model.attention import SelfAttention, CrossAttention, PMA
from model.utils import masked_forward


__all__ = ['SetEncoder', 'GlobalEncoder', 'TaskEncoder', 'ConditionalSetEncoder', 'MultiTaskAttention', 'MTPDecoder',
           'SharedSetEncoder', 'SharedConditionalSetEncoder', 'SharedMTPDecoder']


class SetEncoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers, attn_config):
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers, act_fn=attn_config.act_fn, ln=attn_config.ln)
        self.task_embedding = nn.Parameter(torch.randn(dim_hidden), requires_grad=True)
        self.attention = SelfAttention(dim_hidden, attn_layers, attn_config.n_heads,
                                       act_fn=attn_config.act_fn, ln=attn_config.ln, dr=attn_config.dr)
        self.pool = PMA(dim_hidden, attn_config.n_heads, 1, act_fn=attn_config.act_fn, ln=attn_config.ln, dr=attn_config.dr)
        
    def forward(self, C):
        # nan mask
        mask = C[..., -1].isnan()
        
        # project (x, y) to s
        s = masked_forward(self.mlp, C, mask, self.dim_hidden) # (B, n, h)
        
        # add task embedding e^t
        s = s + self.task_embedding.unsqueeze(0).unsqueeze(1)
        
        # intra-task attention
        s = self.attention(s, mask=mask) # (B, n, h)
            
        # intra-task aggregation
        s = self.pool(s).squeeze(1) # (B, h)
        
        return s


class SharedSetEncoder(nn.Module):
    def __init__(self, n_tasks, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers, attn_config):
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers, act_fn=attn_config.act_fn, ln=attn_config.ln)
        self.task_embedding = nn.Parameter(torch.randn(n_tasks, dim_hidden), requires_grad=True)
        self.attention = SelfAttention(dim_hidden, attn_layers, attn_config.n_heads,
                                       act_fn=attn_config.act_fn, ln=attn_config.ln, dr=attn_config.dr)
        self.pool = PMA(dim_hidden, attn_config.n_heads, 1, act_fn=attn_config.act_fn, ln=attn_config.ln, dr=attn_config.dr)
        
    def forward(self, C):
        # nan mask
        mask = C[..., -1].isnan()
        
        # project (x, y) to s
        s = masked_forward(self.mlp, C, mask, self.dim_hidden) # (B, T, n, h)
        
        # add task embedding e^t
        s = s + self.task_embedding.unsqueeze(0).unsqueeze(2)
        
        # intra-task attention
        B, T = s.size()[:2]
        s = s.view(-1, *s.size()[2:])
        mask = mask.view(-1, *mask.size()[2:])
        s = self.attention(s, mask=mask) # (B*T, n, h)
            
        # intra-task aggregation
        s = self.pool(s).view(B, T, s.size(-1)) # (B, T, h)
        
        return s
    
    
class GlobalEncoder(nn.Module):
    def __init__(self, dim_hidden, attn_layers, attn_config):
        super().__init__()
        self.attention = SelfAttention(dim_hidden, attn_layers, attn_config.n_heads,
                                       act_fn=attn_config.act_fn, ln=attn_config.ln, dr=attn_config.dr)
        self.pool = PMA(dim_hidden, attn_config.n_heads, 1, act_fn=attn_config.act_fn, ln=attn_config.ln, dr=attn_config.dr)
        
        self.global_amortizer = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2, attn_config.act_fn, attn_config.ln)

    def forward(self, s):
        # inter-task attention
        s = self.attention(s) # (B, T, h)
        
        # inter-task aggregation
        s = self.pool(s).squeeze(1) # (B, h)
        
        # global latent distribution
        q_G = self.global_amortizer(s)
        
        return q_G
    
    
class TaskEncoder(nn.Module):
    def __init__(self, dim_hidden, attn_config, hierarchical=True):
        super().__init__()
        self.hierarchical = hierarchical
        self.task_amortizer = LatentMLP(dim_hidden*(1 + int(hierarchical)), dim_hidden, dim_hidden,
                                        2, attn_config.act_fn, attn_config.ln)
        
    def forward(self, s, z=None):
        # hierarchical conditioning
        if self.hierarchical:
            assert z is not None
            s = torch.cat((s, z), -1)
            
        # task latent distribution
        q_T = self.task_amortizer(s)
        
        return q_T
    
    
class ConditionalSetEncoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers, attn_config):
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers, act_fn=attn_config.act_fn, ln=attn_config.ln)
        self.task_embedding = nn.Parameter(torch.randn(dim_hidden), requires_grad=True)
        self.attention = CrossAttention(dim_x, dim_x, dim_hidden, attn_layers, attn_config.n_heads,
                                        attn_config.act_fn, attn_config.ln, attn_config.dr)
        
    def forward(self, C, X_C, X_D):
        # nan mask
        mask = C[..., -1].isnan()
        
        # project (x, y) to s
        d = masked_forward(self.mlp, C, mask, self.dim_hidden) # (B, n, h)
        
        # add task embedding e^t
        d = d + self.task_embedding.unsqueeze(0).unsqueeze(1)
        
        # intra-task attention
        u = self.attention(X_D, X_C, d, mask_K=mask)
        
        return u
    
    
class SharedConditionalSetEncoder(nn.Module):
    def __init__(self, n_tasks, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers, attn_config):
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers, act_fn=attn_config.act_fn, ln=attn_config.ln)
        self.task_embedding = nn.Parameter(torch.randn(n_tasks, dim_hidden), requires_grad=True)
        self.attention = CrossAttention(dim_x, dim_x, dim_hidden, attn_layers, attn_config.n_heads,
                                        attn_config.act_fn, attn_config.ln, attn_config.dr)
        
    def forward(self, C, X_C, X_D):
        # nan mask
        mask = C[..., -1].isnan()
        
        # project (x, y) to s
        d = masked_forward(self.mlp, C, mask, self.dim_hidden) # (B, T, n, h)
        
        # add task embedding e^t
        d = d + self.task_embedding.unsqueeze(0).unsqueeze(2)
        
        # intra-task attention
        B, T = d.size()[:2]
        d = d.view(B*T, *d.size()[2:])
        mask = mask.view(B*T, *mask.size()[2:])
        X_C = X_C.unsqueeze(1).repeat(1, T, 1, 1).view(B*T, *X_C.size()[1:])
        X_D = X_D.unsqueeze(1).repeat(1, T, 1, 1).view(B*T, *X_D.size()[1:])
        u = self.attention(X_D, X_C, d, mask_K=mask)
        u = u.view(B, T, *u.size()[1:])
        
        return u

    
class MultiTaskAttention(nn.Module):
    def __init__(self, dim_hidden, n_layers, n_heads, act_fn='relu', ln=False, dr=0.1):
        super().__init__()
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU
        
        self.attention = SelfAttention(dim_hidden, n_layers, n_heads, act_fn=act_fn, ln=ln, dr=dr)
        
    def forward(self, Q):
        bs, nb, ts, _ = Q.size()
        Q_ = Q.transpose(1, 2).reshape(bs*ts, nb, -1)
        Q_ = self.attention(Q_) # (bs*ts, nb, dim_hidden) or (bs*ts, 1, dim_hidden)
        Q = Q_.reshape(bs, ts, *Q_.size()[1:]).transpose(1, 2)
        return Q
    
    
class MTPDecoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, n_layers, attn_config, sigma):
        super().__init__()
        self.input_projection = nn.Linear(dim_x, dim_hidden)
        self.task_embedding = nn.Parameter(torch.randn(dim_hidden), requires_grad=True)
        self.output_amortizer = LatentMLP(dim_hidden*3, dim_y, dim_hidden, n_layers,
                                          attn_config.act_fn, attn_config.ln, sigma=sigma, sigma_act=F.softplus)
        
    def forward(self, X, v, r):
        # project x to w
        w = self.input_projection(X) # (B, n, h) or (B, ns, n, h)
        
        if self.training:
            # add task embedding e^t
            w = w + self.task_embedding.unsqueeze(0).unsqueeze(1)
        else:
            # add task embedding e^t
            w = w + self.task_embedding.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            
        # concat w, v, r
        v = v.unsqueeze(-2).repeat(*([1]*(len(w.size())-2)), w.size(-2), 1)
        
        decoder_input = torch.cat((w, v, r), -1)
        
        # output distribution
        p_Y = self.output_amortizer(decoder_input)
        
        return p_Y
    
    
class SharedMTPDecoder(nn.Module):
    def __init__(self, n_tasks, dim_x, dim_y, dim_hidden, n_layers, attn_config, sigma):
        super().__init__()
        self.input_projection = nn.Linear(dim_x, dim_hidden)
        self.task_embedding = nn.Parameter(torch.randn(n_tasks, dim_hidden), requires_grad=True)
        self.output_amortizer = LatentMLP(dim_hidden*3, dim_y, dim_hidden, n_layers,
                                          attn_config.act_fn, attn_config.ln, sigma=sigma, sigma_act=F.softplus)
        
    def forward(self, X, v, r):
        # project x to w
        w = self.input_projection(X) # (B, n, h) or (B, ns, n, h)
        w = w.unsqueeze(1).repeat(1, v.size(1), *([1]*(len(w.size())-1))) # (B, T, n, h) or (B, T, ns, n, h)
        
        if self.training:
            # add task embedding e^t
            w = w + self.task_embedding.unsqueeze(0).unsqueeze(2)
        else:
            # add task embedding e^t
            w = w + self.task_embedding.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            
        # concat w, v, r
        v = v.unsqueeze(-2).repeat(*([1]*(len(w.size())-2)), w.size(-2), 1)
        
        decoder_input = torch.cat((w, v, r), -1)
        
        # output distribution
        p_Y = self.output_amortizer(decoder_input)
        
        return p_Y