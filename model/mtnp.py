import torch
import torch.nn as nn
from torch.distributions import Normal

from model.module import *


class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tasks = config.tasks
        
        # latent encoding path
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                     config.module_sizes[0], config.module_sizes[1], config.attn_config)
                                          for task in self.tasks])
        self.global_encoder = GlobalEncoder(config.dim_hidden, config.module_sizes[2], config.attn_config)
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, config.attn_config, hierarchical=True)
                                           for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = nn.ModuleList([ConditionalSetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                                            config.module_sizes[0], config.module_sizes[1], config.attn_config)
                                                      for task in self.tasks])
        self.deterministic_encoder = MultiTaskAttention(config.dim_hidden, config.module_sizes[2], config.attn_config.n_heads,
                                                        config.attn_config.act_fn, config.attn_config.ln, config.attn_config.dr)
        
        # decoding path
        self.decoder = nn.ModuleList([MTPDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3],
                                                 config.attn_config, sigma=(config.task_types[task] == 'continuous'))
                                      for task in self.tasks])
        
    def state_dict_(self):
        return self.state_dict()
    
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)
        
    def encode_global(self, X, Y):
        s = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks):
            D_t = torch.cat((X, Y[task]), -1)
            s[task] = self.set_encoder[t_idx](D_t)
            
        # global latent in across-task inference of latent path
        s_G = torch.stack([s[task] for task in s], 1)
        q_G = self.global_encoder(s_G)
        return q_G, s
    
    def encode_task(self, s, z):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks):
            s_t = s[task]
            if not self.training:
                s_t = s_t.unsqueeze(1).repeat(1, z.size(1), 1)
            q_T[task] = self.task_encoder[t_idx](s_t, z)
            
        return q_T
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        U_C = {}
        # cross-attention layers in across-task inference of deterministic path
        for t_idx, task in enumerate(self.tasks):
            C_t = torch.cat((X_C, Y_C[task]), -1)
            U_C[task] = self.conditional_set_encoder[t_idx](C_t, X_C, X_D)

        # self-attention layers in across-task inference of deterministic path
        U_C = torch.stack([U_C[task] for task in self.tasks], 1)
        r = self.deterministic_encoder(U_C)
        return r
    
    def decode(self, X, v, r):
        if not self.training:
            X = X.unsqueeze(1).repeat(1, v.size(2), 1, 1)
            r = r.unsqueeze(2).repeat(1, 1, v.size(2), 1, 1)
            
        p_Y = {}
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X, v[:, t_idx], r[:, t_idx])
        return p_Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, ns_G=5, ns_T=5):
        if self.training:
            assert Y_D is not None
            
            q_C_G, s_C = self.encode_global(X_C, Y_C)
            q_D_G, s_D = self.encode_global(X_D, Y_D)
            z = Normal(*q_D_G).rsample()

            q_C_T = self.encode_task(s_C, z)
            q_D_T = self.encode_task(s_D, z)
            v = torch.stack([Normal(*q_D_T[task]).rsample() for task in self.tasks], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)

            p_Y = self.decode(X_D, v, r)

            return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
        else:
            q_C_G, s_C = self.encode_global(X_C, Y_C)
            if MAP:
                z = q_C_G[0].unsqueeze(1)
            else:
                z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)
                
            q_C_T = self.encode_task(s_C, z)
            if MAP:
                v = torch.stack([q_C_T[task][0] for task in q_C_T], 1)
            else:
                v = torch.stack([Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1).reshape(z.size(0), ns_G*ns_T, -1)
                                 for task in q_C_T], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)
            
            p_Y = self.decode(X_D, v, r)

            return p_Y

        
class STP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tasks = config.tasks
        
        # latent encoding path
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                     config.module_sizes[0], config.module_sizes[1], config.attn_config)
                                          for task in self.tasks])
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, config.attn_config, hierarchical=False)
                                           for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = nn.ModuleList([ConditionalSetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                                            config.module_sizes[0], config.module_sizes[1], config.attn_config)
                                                      for task in self.tasks])
        
        # decoding path
        self.decoder = nn.ModuleList([MTPDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3],
                                                 config.attn_config, sigma=(config.task_types[task] == 'continuous'))
                                      for task in self.tasks])
        
    def state_dict_(self):
        state_dict = {task: {} for task in self.tasks}
        for name, child in self.named_children():
            for t_idx, task in enumerate(self.tasks):
                state_dict[task][name] = child[t_idx].state_dict()
                
        return state_dict
    
    def state_dict_task(self, task):
        state_dict = {}
        for name, child in self.named_children():
            t_idx = self.tasks.index(task)
            state_dict[name] = child[t_idx].state_dict()
                
        return state_dict
    
    def load_state_dict_(self, state_dict):
        for name, child in self.named_children():
            for t_idx, task in enumerate(self.tasks):
                child[t_idx].load_state_dict(state_dict[task][name])
    
    def encode_task(self, X, Y):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks):
            D_t = torch.cat((X, Y[task]), -1)
            s_t = self.set_encoder[t_idx](D_t)
            q_T[task] = self.task_encoder[t_idx](s_t)
            
        return q_T
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        U_C = {}
        # cross-attention layers in across-task inference of deterministic path
        for t_idx, task in enumerate(self.tasks):
            C_t = torch.cat((X_C, Y_C[task]), -1)
            U_C[task] = self.conditional_set_encoder[t_idx](C_t, X_C, X_D)

        # self-attention layers in across-task inference of deterministic path
        r = torch.stack([U_C[task] for task in self.tasks], 1)
        return r
    
    def decode(self, X, v, r):
        if not self.training:
            X = X.unsqueeze(1).repeat(1, v.size(2), 1, 1)
            r = r.unsqueeze(2).repeat(1, 1, v.size(2), 1, 1)
            
        p_Y = {}
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X, v[:, t_idx], r[:, t_idx])
        return p_Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, ns_G=5, ns_T=5):
        if self.training:
            assert Y_D is not None
            q_C_T = self.encode_task(X_C, Y_C)
            q_D_T = self.encode_task(X_D, Y_D)
            v = torch.stack([Normal(*q_D_T[task]).rsample() for task in self.tasks], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)

            p_Y = self.decode(X_D, v, r)

            return p_Y, None, None, q_D_T, q_C_T
        else:
            q_C_T = self.encode_task(X_C, Y_C)
            if MAP:
                v = torch.stack([q_C_T[task][0].unsqueeze(1) for task in q_C_T], 1)
            else:
                v = torch.stack([Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1)
                                 for task in q_C_T], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)

            p_Y = self.decode(X_D, v, r)

            return p_Y
        

class JTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tasks = config.tasks
        self.dim_ys = config.dim_ys
        self.task_types = config.task_types
        
        dim_y = sum([self.dim_ys[task] for task in self.tasks])
        task_type = 'continuous' if sum([int(config.task_types[task] == 'continuous') for task in self.tasks]) > 0 else 'discrete'
        
        # latent encoding path
        self.set_encoder = SetEncoder(config.dim_x, dim_y, config.dim_hidden,
                                      config.module_sizes[0], config.module_sizes[1], config.attn_config)
        self.global_encoder = TaskEncoder(config.dim_hidden, config.attn_config, hierarchical=False)

        # deterministic encoding path
        self.conditional_set_encoder = ConditionalSetEncoder(config.dim_x, dim_y, config.dim_hidden,
                                                             config.module_sizes[0], config.module_sizes[1], config.attn_config)
        
        # decoding path
        self.decoder = MTPDecoder(config.dim_x, dim_y, config.dim_hidden, config.module_sizes[3],
                                  config.attn_config, sigma=(task_type == 'continuous'))
        
    def state_dict_(self):
        return self.state_dict()
    
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)
        
    def encode_global(self, X, Y):
        # global inference of latent path
        D = torch.cat((X, Y), -1)
        s = self.set_encoder(D)
            
        # global latent in across-task inference of latent path
        q_G = self.global_encoder(s)
        return q_G
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        # cross-attention layers in across-task inference of deterministic path
        C = torch.cat((X_C, Y_C), -1)
        r = self.conditional_set_encoder(C, X_C, X_D)
        return r
    
    def decode(self, X, z, r):
        if not self.training:
            X = X.unsqueeze(1).repeat(1, z.size(1), 1, 1)
            r = r.unsqueeze(1).repeat(1, z.size(1), 1, 1)
            
        p_Y = self.decoder(X, z, r)
        return p_Y
    
    def gather_outputs(self, Y):
        Y = torch.cat([Y[task] for task in self.tasks], -1)
        return Y
    
    def ungather_dists(self, p_Y):
        p_Y_ = {}
        offset = 0
        for task in self.tasks:
            if self.task_types[task] == 'continuous':
                p_Y_[task] = (p_Y[0][..., offset:offset+self.dim_ys[task]],
                              p_Y[1][..., offset:offset+self.dim_ys[task]])
            else:
                p_Y_[task] = p_Y[0][..., offset:offset+self.dim_ys[task]]
            
            offset += self.dim_ys[task]
            
        return p_Y_
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, ns_G=5, ns_T=5):
        if self.training:
            assert Y_D is not None
            
            Y_C = self.gather_outputs(Y_C)
            Y_D = self.gather_outputs(Y_D)
            
            q_C_G = self.encode_global(X_C, Y_C)
            q_D_G = self.encode_global(X_D, Y_D)
            z = Normal(*q_D_G).rsample()

            r = self.encode_deterministic(X_C, Y_C, X_D)

            p_Y = self.decode(X_D, z, r)
            p_Y = self.ungather_dists(p_Y)

            return p_Y, q_D_G, q_C_G, None, None
        else:
            Y_C = self.gather_outputs(Y_C)
            
            q_C_G = self.encode_global(X_C, Y_C)
            if MAP:
                z = q_C_G[0].unsqueeze(1)
            else:
                z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)
            
            p_Y = self.decode(X_D, z, r)
            p_Y = self.ungather_dists(p_Y)

            return p_Y


class SharedMTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert len(set(config.dim_ys.values())) == 1
        assert len(set(config.task_types.values())) == 1
        dim_y = config.dim_ys[config.tasks[0]]
        task_type = config.task_types[config.tasks[0]]
        
        self.tasks = config.tasks
        self.task_type = task_type
        
        # latent encoding path
        self.set_encoder = SharedSetEncoder(len(self.tasks), config.dim_x, dim_y, config.dim_hidden,
                                            config.module_sizes[0], config.module_sizes[1], config.attn_config)
        self.global_encoder = GlobalEncoder(config.dim_hidden, config.module_sizes[2], config.attn_config)
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, config.attn_config, hierarchical=True)
                                           for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = SharedConditionalSetEncoder(len(self.tasks), config.dim_x, dim_y, config.dim_hidden,
                                                                   config.module_sizes[0], config.module_sizes[1], config.attn_config)
        self.deterministic_encoder = MultiTaskAttention(config.dim_hidden, config.module_sizes[2], config.attn_config.n_heads,
                                                        config.attn_config.act_fn, config.attn_config.ln, config.attn_config.dr)
        
        # decoding path
        self.decoder = SharedMTPDecoder(len(self.tasks), config.dim_x, dim_y, config.dim_hidden, config.module_sizes[3],
                                        config.attn_config, sigma=(task_type == 'continuous'))
        
    def state_dict_(self):
        return self.state_dict()
    
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)
        
    def encode_global(self, X, Y):
        # per-task inference of latent path
        D = torch.stack([torch.cat((X, Y[task]), -1) for task in Y], 1)
        s = self.set_encoder(D)
            
        # global latent in across-task inference of latent path
        q_G = self.global_encoder(s)
        return q_G, s
    
    def encode_task(self, s, z):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks):
            s_t = s[:, t_idx]
            if not self.training:
                s_t = s_t.unsqueeze(1).repeat(1, z.size(1), 1)
            q_T[task] = self.task_encoder[t_idx](s_t, z)
            
        return q_T
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        U_C = {}
        # cross-attention layers in across-task inference of deterministic path
        
        C = torch.stack([torch.cat((X_C, Y_C[task]), -1) for task in Y_C], 1)
        U_C = self.conditional_set_encoder(C, X_C, X_D)

        # self-attention layers in across-task inference of deterministic path
        r = self.deterministic_encoder(U_C)
        return r
    
    def decode(self, X, v, r):
        if not self.training:
            X = X.unsqueeze(1).repeat(1, v.size(2), 1, 1)
            r = r.unsqueeze(2).repeat(1, 1, v.size(2), 1, 1)
            
        p_Y = self.decoder(X, v, r)
        
        return p_Y
    
    def ungather_dists(self, p_Y):
        p_Y_ = {}
        for t_idx, task in enumerate(self.tasks):
            if self.task_type == 'continuous':
                p_Y_[task] = (p_Y[0][:, t_idx],
                              p_Y[1][:, t_idx])
            else:
                p_Y_[task] = p_Y[0][:, t_idx]
            
        return p_Y_
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, ns_G=5, ns_T=5):
        if self.training:
            assert Y_D is not None
            
            q_C_G, s_C = self.encode_global(X_C, Y_C)
            q_D_G, s_D = self.encode_global(X_D, Y_D)
            z = Normal(*q_D_G).rsample()

            q_C_T = self.encode_task(s_C, z)
            q_D_T = self.encode_task(s_D, z)
            v = torch.stack([Normal(*q_D_T[task]).rsample() for task in self.tasks], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)

            p_Y = self.decode(X_D, v, r)
            p_Y = self.ungather_dists(p_Y)

            return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
        else:
            q_C_G, s_C = self.encode_global(X_C, Y_C)
            if MAP:
                z = q_C_G[0].unsqueeze(1)
            else:
                z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)
                
            q_C_T = self.encode_task(s_C, z)
            if MAP:
                v = torch.stack([q_C_T[task][0] for task in q_C_T], 1)
            else:
                v = torch.stack([Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1).reshape(z.size(0), ns_G*ns_T, -1)
                                 for task in q_C_T], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)
            
            p_Y = self.decode(X_D, v, r)
            p_Y = self.ungather_dists(p_Y)
            
            return p_Y