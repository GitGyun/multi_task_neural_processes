import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *
  
        
class STP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, ln, n_attn_heads, epsilon,
                 module_sizes, stochastic_path=True, deterministic_path=True, *null_args):
        '''
        description:
            STP model
        
        arguments:
            dim_x:              input space dimension
                                [Type] int

            dim_ys:             output space dimensions
                                [Type] dictionary: dim_ys[task] is int

            dim_hidden:         hidden dimension
                                [Type] int

            tasks:              names of task classes
                                [Type] list of str

            ln:                 wheter to apply layernorm in attentions
                                [Type] bool

            n_attn_heads:       number of heads in attentions
                                [Type] int

            module_sizes:       number of layers for each module
                                [Type] tuple of int, of length 3

            stochastic_path:    wheter to employ stochastic path
                                [Type] bool

            deterministic_path: wheter to employ deterministic path
                                [Type] bool
        '''
        super().__init__()
        self.tasks = tasks
        assert stochastic_path or deterministic_path
        self.stochastic_path = stochastic_path
        self.deterministic_path = deterministic_path
        
        # stochastic encoder
        if self.stochastic_path:
            self.feature_extractor_s = nn.ModuleList([
                STEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln, pool=True)
                for task in tasks
            ])
            self.task_latent_encoder = nn.ModuleList([
                NormalEncoder(dim_hidden, dim_hidden, epsilon)
                for task in tasks
            ])
        
        # deterministic encoder
        if self.deterministic_path:
            self.feature_extractor_d = nn.ModuleList([
                STEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln)
                for task in tasks
            ])
            self.attention_module = nn.ModuleList([
                SingleTaskAttention(dim_x, dim_x, dim_hidden, n_attn_heads, module_sizes[1], ln=ln)
                for task in tasks
            ])
        
        # decoder
        self.decoder = nn.ModuleList([
            Decoder(dim_x, dim_hidden, dim_ys[task], module_sizes[2],
                    stochastic_path, deterministic_path, normal=(task != 'segment'))
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
                mask_C = C[..., -1].isnan()
            
                # stochastic path for prior and posterior
                if self.stochastic_path:
                    S_C = self.feature_extractor_s[t](C, mask_C)
                    q_C[task] = self.task_latent_encoder[t](S_C)
                
                    S_D = self.feature_extractor_s[t](D)
                    q_D[task] = self.task_latent_encoder[t](S_D)
                    v = q_D[task].rsample()
                else:
                    q_D = q_C = v = None
                
                # deterministic path
                if self.deterministic_path:
                    U_C = self.feature_extractor_d[t](C, mask_C)
                    r_D = self.attention_module[t](X_D, X_C, U_C, mask_K=mask_C)
                else:
                    r_D = None
                
                # decoding
                p_Y[task] = self.decoder[t](X_D, v, r_D)
            
            return p_Y, None, None, q_D, q_C
        else:
            if not self.stochastic_path:
                L = 1
                
            p_Ys = [{} for _ in range(L)]
            for t, task in enumerate(self.tasks):
                # prepare context
                C = torch.cat((X_C, Y_C[task]), -1)
                mask_C = C[..., -1].isnan()
            
                # stochastic path
                if self.stochastic_path:
                    S_C = self.feature_extractor_s[t](C, mask_C)
                    q_C = self.task_latent_encoder[t](S_C)
                
                # deterministic path
                if self.deterministic_path:
                    U_C = self.feature_extractor_d[t](C, mask_C)
                    r_D = self.attention_module[t](X_D, X_C, U_C, mask_K=mask_C)
                else:
                    r_D = None
                
                # inference
                for l in range(L):
                    if self.stochastic_path:
                        if MAP:
                            v = q_C.mean
                        else:
                            v = q_C.sample()
                    else:
                        v = None
                        
                    # decoding
                    p_Ys[l][task] = self.decoder[t](X_D, v, r_D)
                    
            return p_Ys
        
        
class JTP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, ln, n_attn_heads, epsilon,
                 module_sizes, stochastic_path=True, deterministic_path=True, *null_args):
        '''
        description:
            JTP model
        
        arguments:
            dim_x:              input space dimension
                                [Type] int

            dim_ys:             output space dimensions
                                [Type] dictionary: dim_ys[task] is int

            dim_hidden:         hidden dimension
                                [Type] int

            tasks:              names of task classes
                                [Type] list of str

            ln:                 wheter to apply layernorm in attentions
                                [Type] bool

            n_attn_heads:       number of heads in attentions
                                [Type] int

            module_sizes:       number of layers for each module
                                [Type] tuple of int, of length 3

            stochastic_path:    wheter to employ stochastic path
                                [Type] bool

            deterministic_path: wheter to employ deterministic path
                                [Type] bool
        '''
        super().__init__()
        dim_y = sum([dim_ys[task] for task in tasks])
        self.dim_ys = dim_ys
        self.tasks = tasks
        assert stochastic_path or deterministic_path
        self.stochastic_path = stochastic_path
        self.deterministic_path = deterministic_path
        
        # stochastic encoder
        if self.stochastic_path:
            self.feature_extractor_s = STEncoder(dim_x, dim_y, dim_hidden,
                                                 n_attn_heads, module_sizes[0], ln=ln, pool=True)
            self.global_latent_encoder = NormalEncoder(dim_hidden, dim_hidden, epsilon)
        
        # deterministic encoder
        if self.deterministic_path:
            self.feature_extractor_d = STEncoder(dim_x, dim_y, dim_hidden,
                                                 n_attn_heads, module_sizes[0], ln=ln)
            self.attention_module = SingleTaskAttention(dim_x, dim_x, dim_hidden,
                                                        n_attn_heads, module_sizes[1], ln=ln)
        
        # decoder
        self.decoder = JTPDecoder(dim_x, dim_hidden, dim_y, module_sizes[2],
                                  stochastic_path, deterministic_path)
        
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
            
            # stochastic path for prior and posterior
            if self.stochastic_path:
                S_C = self.feature_extractor_s(C)
                q_C_G = self.global_latent_encoder(S_C)
            
                S_D = self.feature_extractor_s(D)
                q_D_G = self.global_latent_encoder(S_D)
                z = q_D_G.rsample()
            else:
                q_C_G = q_D_G = z = None

            # deterministic path
            if self.deterministic_path:
                U_C = self.feature_extractor_d(C)
                r_D = self.attention_module(X_D, X_C, U_C)
            else:
                r_D = None

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
            if self.stochastic_path:
                S_C = self.feature_extractor_s(C)
                q_C_G = self.global_latent_encoder(S_C)
            
            # deterministic path
            if self.deterministic_path:
                U_C = self.feature_extractor_d(C)
                r_D = self.attention_module(X_D, X_C, U_C)
            else:
                r_D = None
            
            # inference
            p_Ys = [{} for _ in range(K)]
            for k in range(K):
                if self.stochastic_path:
                    if MAP:
                        z = q_C_G.mean
                    else:
                        z = q_C_G.sample()
                else:
                    z = None
                
                # decoding
                p_Y_ = self.decoder(X_D, z, r_D)
                offset = 0
                for t, task in enumerate(self.tasks):
                    p_Ys[k][task] = Normal(p_Y_.mean[..., offset:offset+self.dim_ys[task]],
                                           p_Y_.stddev[..., offset:offset+self.dim_ys[task]])
                    offset += self.dim_ys[task]
            
            return p_Ys


class MTP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, ln, n_attn_heads, epsilon,
                 module_sizes, stochastic_path=True, deterministic_path=True,
                 implicit_global_latent=False, global_latent_only=False,
                 deterministic_path2=False, context_posterior=False):
        '''
        description:
            MTP model
        
        arguments:
            dim_x:                   input space dimension
                                     [Type] int

            dim_ys:                  output space dimensions
                                     [Type] dictionary: dim_ys[task] is int

            dim_hidden:              hidden dimension
                                     [Type] int

            tasks:                   names of task classes
                                     [Type] list of str

            ln:                      wheter to apply layernorm in attentions
                                     [Type] bool

            n_attn_heads:            number of heads in attentions
                                     [Type] int

            module_sizes:            number of layers for each module
                                     [Type] tuple of int, of length 4

            stochastic_path:         wheter to employ stochastic path
                                     [Type] bool

            deterministic_path:      wheter to employ deterministic path
                                     [Type] bool

            implicit_global_latent:  wheter to implicitly specify global latent in stochastic path
                                     [Type] bool

            global_latent_only:      wheter to use only global latent (without per-task latents)
                                     [Type] bool
        '''
        super().__init__()
        self.tasks = tasks
        assert stochastic_path or deterministic_path or deterministic_path2
        assert not (stochastic_path and deterministic_path2)
        self.stochastic_path = stochastic_path
        self.deterministic_path = deterministic_path
        self.implicit_global_latent = implicit_global_latent
        self.global_latent_only = global_latent_only
        self.deterministic_path2 = deterministic_path2
        self.context_posterior = context_posterior
        
        # aggregation modules
        if self.stochastic_path:
            self.feature_extractor_s = nn.ModuleList([
                STEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads,
                          module_sizes[0], ln=ln, pool=True)
                for task in tasks
            ])
            
            if self.implicit_global_latent:
                self.global_attention = nn.Sequential(
                    *[SAB(dim_hidden, n_attn_heads, ln=ln) for _ in range(module_sizes[3])],
                )
                
                self.task_latent_encoder = nn.ModuleList([
                    NormalEncoder(dim_hidden, dim_hidden, epsilon)
                    for task in tasks
                ])
            else:
                self.global_attention = nn.Sequential(
                    *[SAB(dim_hidden, n_attn_heads, ln=ln) for _ in range(module_sizes[3])],
                    PMA(dim_hidden, n_attn_heads, 1, ln=ln),
                    Squeeze(1),
                )
                self.global_latent_encoder = NormalEncoder(dim_hidden, dim_hidden, epsilon)
                
                if not self.global_latent_only:
                    self.task_latent_encoder = nn.ModuleList([
                        NormalEncoder(2*dim_hidden, dim_hidden, epsilon)
                        for task in tasks
                    ])
                    
        elif self.deterministic_path2:
            self.feature_extractor_s = nn.ModuleList([
                STEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads,
                          module_sizes[0], ln=ln, pool=True)
                for task in tasks
            ])
            
            if self.implicit_global_latent:
                self.global_attention = nn.Sequential(
                    *[SAB(dim_hidden, n_attn_heads, ln=ln) for _ in range(module_sizes[3])],
                )
                
                self.task_encoder = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden),
                        nn.ReLU(),
                        nn.Linear(dim_hidden, dim_hidden)
                    )
                    for task in tasks
                ])
            else:
                self.global_attention = nn.Sequential(
                    *[SAB(dim_hidden, n_attn_heads, ln=ln) for _ in range(module_sizes[3])],
                    PMA(dim_hidden, n_attn_heads, 1, ln=ln),
                    Squeeze(1),
                )
                
                self.global_encoder = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.ReLU(),
                    nn.Linear(dim_hidden, dim_hidden)
                )
                
                if not self.global_latent_only:
                    self.task_encoder = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(2*dim_hidden, dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden, dim_hidden)
                        )
                        for task in tasks
                    ])


        # attention modules
        if self.deterministic_path:
            self.feature_extractor_d = nn.ModuleList([
                STEncoder(dim_x, dim_ys[task], dim_hidden, n_attn_heads, module_sizes[0], ln=ln)
                for task in tasks
            ])

            self.attention_module = MultiTaskAttention(tasks, dim_x, dim_x, dim_hidden,
                                                       n_attn_heads, module_sizes[1], ln=ln)
        
        # decoder
        self.decoder = nn.ModuleList([
            Decoder(dim_x, dim_hidden, dim_ys[task], module_sizes[2],
                    stochastic_path or deterministic_path2, deterministic_path, normal=(task != 'segment'))
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
                masks_C.append(C[..., -1].isnan())
                if self.stochastic_path and not self.context_posterior:
                    D = torch.cat((X_D, Y_D[task]), -1)
            
                # stochastic paths
                if self.stochastic_path or self.deterministic_path2:
                    S_C[task] = self.feature_extractor_s[t](C, masks_C[-1])
                    if self.stochastic_path and not self.context_posterior:
                        S_D[task] = self.feature_extractor_s[t](D)
                
                # deterministic path
                if self.deterministic_path:
                    U_C.append(self.feature_extractor_d[t](C, masks_C[-1]))
            
            # global latent encoding
            q_C_G = q_D_G = z = None
            r_D = [None for _ in self.tasks]
            if self.stochastic_path or self.deterministic_path2:
                S_C = torch.stack([S_C[task] for task in self.tasks], 1)
                S_C_G = self.global_attention(S_C)
                
                if self.stochastic_path and not self.context_posterior:
                    S_D = torch.stack([S_D[task] for task in self.tasks], 1)
                    S_D_G = self.global_attention(S_D)
                
                if not self.implicit_global_latent:
                    if self.deterministic_path2:
                        z = self.global_encoder(S_C_G)
                    elif self.context_posterior:
                        q_C_G = self.global_latent_encoder(S_C_G)
                        z = q_C_G.rsample()
                    else:
                        q_C_G = self.global_latent_encoder(S_C_G)
                        q_D_G = self.global_latent_encoder(S_D_G)
                        z = q_D_G.rsample()
            
            # multi-task attention
            if self.deterministic_path:
                U_C = torch.stack(U_C)
                r_D = self.attention_module(X_D, X_C, U_C, masks=masks_C).permute(2, 0, 1, 3)
            
            p_Y = {}
            q_C = {}
            q_D = {}
            for t, task in enumerate(self.tasks):
                # per-task latent encoding
                if self.stochastic_path or self.deterministic_path2:
                    if self.global_latent_only:
                        q_C = q_D = None
                        v = z
                    elif self.implicit_global_latent:
                        if self.deterministic_path2:
                            q_C = q_D = None
                            v = self.task_encoder[t](S_C_G[:, t])
                        elif self.context_posterior:
                            q_C[task] = self.task_latent_encoder[t](S_C_G[:, t])
                            q_D = None
                            v = q_C[task].rsample()
                        else:
                            q_C[task] = self.task_latent_encoder[t](S_C_G[:, t])
                            q_D[task] = self.task_latent_encoder[t](S_D_G[:, t])
                            v = q_D[task].rsample()
                    else:
                        if self.deterministic_path2:
                            q_C = q_D = None
                            v = self.task_encoder[t](torch.cat((S_C[:, t], 1), -1))
                        elif self.context_posterior:
                            q_C[task] = self.task_latent_encoder[t](torch.cat((S_C[:, t], z), -1))
                            v = q_C[task].rsample()
                        else:
                            q_C[task] = self.task_latent_encoder[t](torch.cat((S_C[:, t], z), -1))
                            q_D[task] = self.task_latent_encoder[t](torch.cat((S_D[:, t], z), -1))
                            v = q_D[task].rsample()
                        
                else:
                    q_C = q_D = v = None
                
                # decoding
                p_Y[task] = self.decoder[t](X_D, v, r_D[t])

            return p_Y, q_D_G, q_C_G, q_D, q_C
        else:
            masks_C = []
            S_C = {}
            U_C = []
            for t, task in enumerate(self.tasks):
                # prepare context
                C = torch.cat((X_C, Y_C[task]), -1)
                masks_C.append(C[..., -1].isnan())
            
                # stochastic paths
                if self.stochastic_path or self.deterministic_path2:
                    S_C[task] = self.feature_extractor_s[t](C, masks_C[-1])
                
                # deterministic path
                if self.deterministic_path:
                    U_C.append(self.feature_extractor_d[t](C, masks_C[-1]))
            
            
            # global latent encoding
            if self.stochastic_path or self.deterministic_path2:
                S_C = torch.stack([S_C[task] for task in self.tasks], 1)
                S_C_G = self.global_attention(S_C)
                    
                if not self.implicit_global_latent:
                    if self.deterministic_path2:
                        z = self.global_encoder(S_C_G)
                    else:
                        q_C_G = self.global_latent_encoder(S_C_G)
            
            # multi-task attention
            r_D = [None for _ in self.tasks]
            if self.deterministic_path:
                U_C = torch.stack(U_C)
                r_D = self.attention_module(X_D, X_C, U_C, masks=masks_C).permute(2, 0, 1, 3)
            
            # inference
            if not self.stochastic_path:
                K = L = 1
            if self.implicit_global_latent:
                K = 1
            if self.global_latent_only:
                L = 1
                
            p_Ys = [{} for _ in range(K*L)]
            for k in range(K):
                if self.stochastic_path and not self.implicit_global_latent:
                    if MAP:
                        z = q_C_G.mean
                    else:
                        z = q_C_G.sample()
                
                for t, task in enumerate(self.tasks):
                    for l in range(L):
                        if self.stochastic_path or self.deterministic_path2:
                            if self.global_latent_only:
                                v = z
                            elif self.implicit_global_latent:
                                if self.stochastic_path:
                                    q_C = self.task_latent_encoder[t](S_C_G[:, t])
                                    if MAP:
                                        v = q_C.mean
                                    else:
                                        v = q_C.sample()
                                else:
                                    v = self.task_encoder[t](S_C_G[:, t])
                            else:
                                if self.stochastic_path:
                                    q_C = self.task_latent_encoder[t](torch.cat((S_C[:, t], z), -1))
                                    if MAP:
                                        v = q_C.mean
                                    else:
                                        v = q_C.sample()
                                else:
                                    v = self.task_encoder[t](torch.cat((S_C[:, t], z), -1))
                                    
                        else:
                            v = None

                        # decoding
                        p_Ys[k*L+l][task] = self.decoder[t](X_D, v, r_D[t])

            return p_Ys
