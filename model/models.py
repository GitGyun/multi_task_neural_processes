import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *
  
        
class STP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, ln, n_attn_heads, module_sizes):
        '''
        description:
            STP model
        
        arguments:
            dim_x:        input space dimension
                          [Type] int

            dim_ys:       output space dimensions
                          [Type] dictionary: dim_ys[task] is int

            dim_hidden:   hidden dimension
                          [Type] int

            tasks:        names of task classes
                          [Type] list of str

            ln:           wheter to apply layernorm in attentions
                          [Type] bool

            n_attn_heads: number of heads in attentions
                          [Type] int

            module_sizes: number of layers for each module
                          [Type] tuple of int, of length 3
            '''
        super().__init__()
        self.tasks = tasks
        
        # stochastic encoder
        self.feature_extractor_s = nn.ModuleList([
            SAEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln, pool=True)
            for task in tasks
        ])
        self.task_latent_encoder = nn.ModuleList([
            NormalEncoder(dim_hidden, dim_hidden)
            for task in tasks
        ])
        
        # deterministic encoder
        self.feature_extractor_d = nn.ModuleList([
            SAEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln)
            for task in tasks
        ])
        self.attention_module = nn.ModuleList([
            CAB(dim_x, dim_x, dim_hidden, n_attn_heads, module_sizes[1], ln=ln)
            for task in tasks
        ])
        
        # decoder
        self.decoder = nn.ModuleList([
            NormalDecoder(dim_x, dim_hidden, dim_ys[task], module_sizes[2])
            for task in tasks
        ])
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, K=1, L=1):
        '''
        inputs:
            X_C: context inputs
                 [Type] torch array of size (B, M, dim_x)
                 
            Y_C: context labels
                 [Type] dictionary: Y_C[task] is torch array of size (B, M, dim_ys[task])
                 
            X_D: target inputs
                 [Type] torch array of size (B, N, dim_x)
                 
            Y_D: target labels
                 [Type] dictionary where Y_D[task] is torch array of size (B, N, dim_ys[task])
                 
            MAP: whether to perform MAP estimation at inference
                 [Type] bool
                 
            K:   unused variable (for compatability with MTP)
                 [Type] int
                 
            L:   the number of per-task latent samples at inference
                 [Type] int
            
        outputs:
            p_Y:   predictive distributions
                   [Type] dictionary: p_Y[task] is torch Normal distribution of size (B, M, dim_ys[task])
                   
            q_D_G: not used
                   [Type] None
                   
            q_C_G: not used
                   [Type] None
                   
            q_D:   posterior per-task latent distributions
                   [Type] dictionary: q_D[task] is torch Normal distribution of size (B, dim_hidden)
                   
            q_C:   prior per-task latent distributions
                   [Type] dictionary: q_C[task] is torch Normal distribution of size (B, dim_hidden)
        '''
        if self.training:
            assert Y_D is not None
            
            p_Y = {}
            q_C = {}
            q_D = {}
            for t, task in enumerate(self.tasks):
                # prepare context and target
                C = torch.cat((X_C, Y_C[task]), -1)
                D = torch.cat((X_D, Y_D[task]), -1)
                mask_C = C[..., 1].isnan()
            
                # stochastic path for prior
                S_C = self.feature_extractor_s[t](C, mask_C)
                q_C[task] = self.task_latent_encoder[t](S_C)
                
                # stochastic path for posterior
                S_D = self.feature_extractor_s[t](D)
                q_D[task] = self.task_latent_encoder[t](S_D)
                v = q_D[task].rsample()
                
                # deterministic path
                U_C = self.feature_extractor_d[t](C, mask_C)
                r_D = self.attention_module[t](X_D, X_C, U_C, mask_K=mask_C)
                
                # decoding
                p_Y[task] = self.decoder[t](X_D, v, r_D)
            
            return p_Y, None, None, q_D, q_C
        else:
            p_Ys = [{} for _ in range(L)]
            for t, task in enumerate(self.tasks):
                # prepare context
                C = torch.cat((X_C, Y_C[task]), -1)
                mask_C = C[..., 1].isnan()
            
                # stochastic path
                S_C = self.feature_extractor_s[t](C, mask_C)
                q_C = self.task_latent_encoder[t](S_C)
                
                # deterministic path
                U_C = self.feature_extractor_d[t](C, mask_C)
                r_D = self.attention_module[t](X_D, X_C, U_C, mask_K=mask_C)
                
                # inference
                for l in range(L):
                    if MAP:
                        v = q_C.mean
                    else:
                        v = q_C.sample()
                        
                    # decoding
                    p_Ys[l][task] = self.decoder[t](X_D, v, r_D)
                    
            return p_Ys
        
        
class JTP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, ln, n_attn_heads, module_sizes):
        '''
        description:
            JTP model
            
        arguments:
            dim_x:        input space dimension
                          [Type] int

            dim_ys:       output space dimensions
                          [Type] dictionary: dim_ys[task] is int

            dim_hidden:   hidden dimension
                          [Type] int

            tasks:        names of task classes
                          [Type] list of str

            ln:           wheter to apply layernorm in attentions
                          [Type] bool

            n_attn_heads: number of heads in attentions
                          [Type] int

            module_sizes: number of layers for each module
                          [Type] tuple of int, of length 3
        '''
        super().__init__()
        dim_y = sum([dim_ys[task] for task in tasks])
        self.dim_ys = dim_ys
        self.tasks = tasks
        
        # stochastic encoder
        self.feature_extractor_s = SAEncoder(dim_x, dim_y, dim_hidden, n_attn_heads, module_sizes[0], ln=ln, pool=True)
        self.global_latent_encoder = NormalEncoder(dim_hidden, dim_hidden)
        
        # deterministic encoder
        self.feature_extractor_d = SAEncoder(dim_x, dim_y, dim_hidden, n_attn_heads, module_sizes[0], ln=ln)
        self.attention_module = CAB(dim_x, dim_x, dim_hidden, n_attn_heads, module_sizes[1], ln=ln)
        
        # decoder
        self.decoder = NormalDecoder(dim_x, dim_hidden, dim_y, module_sizes[2])
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, K=1, L=1):
        '''
        inputs:
            X_C: context inputs
                 [Type] torch array of size (B, M, dim_x)
                 
            Y_C: context labels
                 [Type] dictionary: Y_C[task] is torch array of size (B, M, dim_ys[task])
                 
            X_D: target inputs
                 [Type] torch array of size (B, N, dim_x)
                 
            Y_D: target labels
                 [Type] dictionary where Y_D[task] is torch array of size (B, N, dim_ys[task])
                 
            MAP: whether to perform MAP estimation at inference
                 [Type] bool
                 
            K:   the number of global latent samples at inference
                 [Type] int
                 
            L:   unused variable (for compatability with MTP)
                 [Type] int
            
        outputs:
            p_Y:   predictive distributions
                   [Type] dictionary: p_Y[task] is torch Normal distribution of size (B, M, dim_ys[task])
                   
            q_D_G: posterior global latent distributions
                   [Type] torch Normal distribution of size (B, dim_hidden)
                   
            q_C_G: prior global latent distributions
                   [Type] torch Normal distribution of size (B, dim_hidden)
                   
            q_D:   not used
                   [Type] None
                   
            q_C:   not used
                   [Type] None
        '''
        if self.training:
            assert Y_D is not None
            
            # prepare context and target
            Y_C = torch.cat([Y_C[task] for task in self.tasks], -1)
            Y_D = torch.cat([Y_D[task] for task in self.tasks], -1)
            C = torch.cat((X_C, Y_C), -1)
            D = torch.cat((X_D, Y_D), -1)
            
            # stochastic path for prior
            S_C = self.feature_extractor_s(C)
            q_C_G = self.global_latent_encoder(S_C)
            
            # stochastic path for posterior
            S_D = self.feature_extractor_s(D)
            q_D_G = self.global_latent_encoder(S_D)
            z = q_D_G.rsample()

            # deterministic path
            U_C = self.feature_extractor_d(C)
            r_D = self.attention_module(X_D, X_C, U_C)

            # decoding
            p_Y = {}
            p_Y_ = self.decoder(X_D, z, r_D)
            offset = 0
            for t, task in enumerate(self.tasks):
                p_Y[task] = Normal(p_Y_.mean[..., offset:offset+self.dim_ys[task]],
                                   p_Y_.stddev[..., offset:offset+self.dim_ys[task]])
                offset += self.dim_ys[task]
            
            return p_Y, q_D_G, q_C_G, None, None
        else:
            # prepare context
            Y_C = torch.cat([Y_C[task] for task in self.tasks], -1)
            C = torch.cat((X_C, Y_C), -1)
            
            # stochastic path
            S_C = self.feature_extractor_s(C)
            q_C_G = self.global_latent_encoder(S_C)
            
            # deterministic path
            U_C = self.feature_extractor_d(C)
            r_D = self.attention_module(X_D, X_C, U_C)
            
            # inference
            p_Ys = [{} for _ in range(K)]
            for k in range(K):
                if MAP:
                    z = q_C_G.mean
                else:
                    z = q_C_G.sample()
                
                # decoding
                p_Y_ = self.decoder(X_D, z, r_D)
                offset = 0
                for t, task in enumerate(self.tasks):
                    p_Ys[k][task] = Normal(p_Y_.mean[..., offset:offset+self.dim_ys[task]],
                                           p_Y_.stddev[..., offset:offset+self.dim_ys[task]])
                    offset += self.dim_ys[task]
            
            return p_Ys


class MTP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, ln, n_attn_heads, module_sizes):
        '''
        description:
            MTP model
            
        arguments:
            dim_x:        input space dimension
                          [Type] int

            dim_ys:       output space dimensions
                          [Type] dictionary: dim_ys[task] is int

            dim_hidden:   hidden dimension
                          [Type] int

            tasks:        names of task classes
                          [Type] list of str

            ln:           wheter to apply layernorm in attentions
                          [Type] bool

            n_attn_heads: number of heads in attentions
                          [Type] int

            module_sizes: number of layers for each module
                          [Type] tuple of int, of length 3
        '''
        super().__init__()
        self.tasks = tasks
        
        # stochastic encoder
        self.feature_extractor_s = nn.ModuleList([
            SAEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln, pool=True)
            for task in tasks
        ])
        self.global_latent_encoder = nn.Sequential(
            *[SAB(dim_hidden, n_attn_heads, ln=ln) for _ in range(module_sizes[0])],
            PMA(dim_hidden, n_attn_heads, 1, ln=ln),
            Squeeze(1),
            NormalEncoder(dim_hidden, dim_hidden)
        )
        self.task_latent_encoder = nn.ModuleList([
            NormalEncoder(2*dim_hidden, dim_hidden)
            for task in tasks
        ])
        
        # deterministic encoder
        self.feature_extractor_d = nn.ModuleList([
            SAEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln)
            for task in tasks
        ])

        self.attention_module = MultiTaskAttention(tasks, dim_x, dim_x, dim_hidden,
                                                   n_attn_heads, module_sizes[1], ln=ln)
#         if multitask_attention:
#             self.attention_module = MultiTaskAttention(tasks, dim_x, dim_x, dim_hidden,
#                                                        n_attn_heads, base_attn_layers, ln=ln)
#         else:
#             self.attention_module = nn.ModuleList([
#                 CAB(dim_x, dim_x, dim_hidden, n_attn_heads, base_attn_layers, ln=ln)
#                 for task in tasks
#             ])
        
        # decoder
        self.decoder = nn.ModuleList([
            NormalDecoder(dim_x, dim_hidden, dim_ys[task], module_sizes[2])
            for task in tasks
        ])
        
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, L=1, K=1):
        '''
        inputs:
            X_C: context inputs
                 [Type] torch array of size (B, M, dim_x)
                 
            Y_C: context labels
                 [Type] dictionary: Y_C[task] is torch array of size (B, M, dim_ys[task])
                 
            X_D: target inputs
                 [Type] torch array of size (B, N, dim_x)
                 
            Y_D: target labels
                 [Type] dictionary where Y_D[task] is torch array of size (B, N, dim_ys[task])
                 
            MAP: whether to perform MAP estimation at inference
                 [Type] bool
                 
            L:   the number of per-task latent samples at inference
                 [Type] int
                 
            K:   the number of global latent samples at inference
                 [Type] int
            
        outputs:
            p_Y:   predictive distributions
                   [Type] dictionary: p_Y[task] is torch Normal distribution of size (B, M, dim_ys[task])
                   
            q_D_G: posterior global latent distributions
                   [Type] torch Normal distribution of size (B, dim_hidden)
                   
            q_C_G: prior global latent distributions
                   [Type] torch Normal distribution of size (B, dim_hidden)
                   
            q_D:   posterior per-task latent distributions
                   [Type] dictionary: q_D[task] is torch Normal distribution of size (B, dim_hidden)
                   
            q_C:   prior per-task latent distributions
                   [Type] dictionary: q_C[task] is torch Normal distribution of size (B, dim_hidden)
        '''
        if self.training:
            assert Y_D is not None
            
            masks_C = []
            S_C = {}
            S_D = {}
            U_C = []
            for t, task in enumerate(self.tasks):
                # prepare context and target
                C = torch.cat((X_C, Y_C[task]), -1)
                D = torch.cat((X_D, Y_D[task]), -1)
                masks_C.append(C[..., 1].isnan())
            
                # stochastic paths
                S_C[task] = self.feature_extractor_s[t](C, masks_C[-1])
                S_D[task] = self.feature_extractor_s[t](D)
                
                # deterministic path
                U_C.append(self.feature_extractor_d[t](C, masks_C[-1]))
            
            # global latent encoding
            S_C = torch.stack([S_C[task] for task in self.tasks], 1)
            q_C_G = self.global_latent_encoder(S_C)
            
            S_D = torch.stack([S_D[task] for task in self.tasks], 1)
            q_D_G = self.global_latent_encoder(S_D)
            z = q_D_G.rsample()
            
            # multi-task attention
            U_C = torch.stack(U_C)
            r_D = self.attention_module(X_D, X_C, U_C, masks=masks_C)
            
            p_Y = {}
            q_C = {}
            q_D = {}
            for t, task in enumerate(self.tasks):
                # stochastic path
                q_C[task] = self.task_latent_encoder[t](torch.cat((S_C[:, t], z), -1))
                q_D[task] = self.task_latent_encoder[t](torch.cat((S_D[:, t], z), -1))
                v = q_D[task].rsample()
                
                # decoding
                p_Y[task] = self.decoder[t](X_D, v, r_D[:, :, t])

            return p_Y, q_D_G, q_C_G, q_D, q_C
        else:
            masks_C = []
            S_C = {}
            U_C = []
            for t, task in enumerate(self.tasks):
                # prepare context
                C = torch.cat((X_C, Y_C[task]), -1)
                masks_C.append(C[..., 1].isnan())
            
                # stochastic paths
                S_C[task] = self.feature_extractor_s[t](C, masks_C[-1])
                
                # deterministic path
                U_C.append(self.feature_extractor_d[t](C, masks_C[-1]))
            
            # global latent encoding
            S_C = torch.stack([S_C[task] for task in self.tasks], 1)
            q_C_G = self.global_latent_encoder(S_C)
            
            # multi-task attention
            U_C = torch.stack(U_C)
            r_D = self.attention_module(X_D, X_C, U_C, masks=masks_C)
            
            # inference
            p_Ys = [{} for _ in range(K*L)]
            for k in range(K):
                if MAP:
                    z = q_C_G.mean
                else:
                    z = q_C_G.sample()
                
                for t, task in enumerate(self.tasks):
                    for l in range(L):
                        q_C = self.task_latent_encoder[t](torch.cat((S_C[:, t], z), -1))
                        if MAP:
                            v = q_C.mean
                        else:
                            v = q_C.sample()

                        # decoding
                        p_Ys[k*L+l][task] = self.decoder[t](X_D, v, r_D[:, :, t])

            return p_Ys
