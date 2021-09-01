import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .modules import MLP, LatentMLP, STEncoder, CrossAttention, MultiTaskAttention
from .utils import masked_forward


def get_model(config, device):
    if config.shared:
        return SharedMultiTaskNP(config.dim_x, config.dim_ys, config.dim_hidden, config.tasks, config.task_blocks_model,
                           config.module_sizes, config.task_latents, config.global_latent,
                           config.stochastic_path, config.deterministic_path, config.local_deterministic_path, config.task_embedding,
                           config.n_attn_heads, config.activation, config.layernorm, config.dropout, config.skip, config.epsilon).to(device)
    else:
        return MultiTaskNP(config.dim_x, config.dim_ys, config.dim_hidden, config.tasks, config.task_blocks_model,
                           config.module_sizes, config.task_latents, config.global_latent,
                           config.stochastic_path, config.deterministic_path, config.local_deterministic_path, config.task_embedding,
                           config.n_attn_heads, config.activation, config.layernorm, config.dropout, config.skip, config.epsilon).to(device)


class DataParallel(nn.DataParallel):
    def state_dict_(self):
        return self.module.state_dict_()
    
    def load_state_dict_(self, state_dict):
        self.module.load_state_dict_(state_dict)


class MultiTaskNP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, task_blocks, module_sizes,
                 task_latents, global_latent, stochastic_path, deterministic_path, local_deterministic_path, task_embedding,
                 n_attn_heads, activation, ln, dr, skip, epsilon):

        super().__init__()
        self.tasks = tasks
        self.dim_ys = dim_ys
        self.dim_hidden = dim_hidden
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise NotImplementedError
        
        # configure tasks and task blocks
        self.dim_y_blocks = {}
        self.block_names = []
        self.task_blocks = {}
        for b, task_block in enumerate(task_blocks):
            block = ','.join(task_block)
            self.block_names.append(block)
            self.task_blocks[block] = task_block
            self.dim_y_blocks[block] = 0
            for task in task_block:
                self.dim_y_blocks[block] += dim_ys[task]
            
        # configure paths
        assert stochastic_path or deterministic_path or local_deterministic_path
        self.stochastic_path = stochastic_path
        self.deterministic_path = deterministic_path
        self.local_deterministic_path = local_deterministic_path
        
        # configure LVM
        assert task_latents or global_latent
        self.task_latents = task_latents
        self.global_latent = global_latent
        
        self.task_embedding = task_embedding
        
        # stochastic path
        if self.stochastic_path:
            self.encoder_s = nn.ModuleList([
                MLP(dim_x + self.dim_y_blocks[block], dim_hidden, dim_hidden, module_sizes[0],
                    act_fn, ln, dr, skip)
                for block in self.block_names
            ])
            
            self.intra_task_attention_s = nn.ModuleList([
                STEncoder(dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)
                for block in self.block_names
            ])

            if self.task_latents:
                dim_in = dim_hidden*2 if self.global_latent else dim_hidden
                self.task_latent_encoder_s = nn.ModuleList([
                    LatentMLP(dim_in, dim_hidden, dim_hidden, 2, act_fn, ln, dr, skip, epsilon=epsilon)
                    for task in tasks
                ])
            
            if self.global_latent:
                self.inter_task_attention_s = STEncoder(dim_hidden, module_sizes[2], n_attn_heads, act_fn, ln, dr)
                self.global_latent_encoder_s = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2,
                                                         act_fn, ln, dr, skip, epsilon=epsilon)
                
                # task embedding
                if self.task_embedding:
                    self.task_embedding_s = nn.Parameter(torch.randn(len(self.block_names), dim_hidden),
                                                         requires_grad=True)
            
        # deterministic path        
        if self.deterministic_path:
            self.encoder_d = nn.ModuleList([
                MLP(dim_x + self.dim_y_blocks[block], dim_hidden, dim_hidden, module_sizes[0],
                    act_fn, ln, dr, skip)
                for block in self.block_names
            ])
            
            self.intra_task_attention_d = nn.ModuleList([
                STEncoder(dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)
                for block in self.block_names
            ])

            if self.task_latents:
                dim_in = dim_hidden*2 if self.global_latent else dim_hidden
                self.task_latent_encoder_d = nn.ModuleList([
                    LatentMLP(dim_in, dim_hidden, dim_hidden, 2, act_fn, ln, dr, skip, sigma=False)
                    for task in tasks
                ])
            
            if self.global_latent:
                self.inter_task_attention_d = STEncoder(dim_hidden, module_sizes[2], n_attn_heads, act_fn, ln, dr)
                self.global_latent_encoder_d = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2,
                                                         act_fn, ln, dr, skip, sigma=False)
                
                # task embedding
                if self.task_embedding:
                    self.task_embedding_d = nn.Parameter(torch.randn(len(self.block_names), dim_hidden),
                                                         requires_grad=True)
        
        # local deterministic path
        if self.local_deterministic_path:
            self.encoder_l = nn.ModuleList([
                MLP(dim_x + self.dim_y_blocks[block], dim_hidden, dim_hidden, module_sizes[0],
                    act_fn, ln, dr, skip)
                for block in self.block_names
            ])
            
            self.intra_task_attention_l = nn.ModuleList([
                CrossAttention(dim_x, dim_x, dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)
                for block in self.block_names
            ])

            if self.global_latent:
                self.inter_task_attention_l = MultiTaskAttention(dim_hidden, module_sizes[2], n_attn_heads,
                                                                 act_fn, ln, dr, pool=(not self.task_latents))
                
                # task embedding
                if self.task_embedding:
                    self.task_embedding_l = nn.Parameter(torch.randn(len(self.block_names), dim_hidden),
                                                         requires_grad=True)
                
        
        # decoder
        self.decoder_head = nn.ModuleList([
            nn.Linear(dim_x, dim_hidden)
            for block in self.block_names
        ])
        
        dim_in = dim_hidden * (1 + int(self.stochastic_path) + int(self.deterministic_path) + int(self.local_deterministic_path))
        self.decoder = nn.ModuleList([
            LatentMLP(dim_in, self.dim_y_blocks[block], dim_hidden, module_sizes[3],
                      act_fn, ln, dr, epsilon=epsilon, sigma_act=F.softplus)
            for block in self.block_names
        ])
        
    def state_dict_global(self):
        state_dict = {}
        
        if self.global_latent:
            if self.stochastic_path:
                state_dict['inter_task_attention_s'] = self.inter_task_attention_s.state_dict()
                state_dict['global_latent_encoder_s'] = self.global_latent_encoder_s.state_dict()
                
                if self.task_embedding:
                    state_dict['task_embedding_s'] = self.task_embedding_s.data
                
            if self.deterministic_path:
                state_dict['inter_task_attention_d'] = self.inter_task_attention_d.state_dict()
                state_dict['global_latent_encoder_d'] = self.global_latent_encoder_d.state_dict()
                
                if self.task_embedding:
                    state_dict['task_embedding_d'] = self.task_embedding_d.data
                
            if self.local_deterministic_path:
                state_dict['inter_task_attention_l'] = self.inter_task_attention_l.state_dict()
                
                if self.task_embedding:
                    state_dict['task_embedding_l'] = self.task_embedding_l.data
                
        
        return state_dict
                
                
    def state_dict_block(self, block):
        b_idx = self.block_names.index(block)
        state_dict = {}
        
        if self.stochastic_path:
            state_dict['encoder_s'] = self.encoder_s[b_idx].state_dict()
            state_dict['intra_task_attention_s'] = self.intra_task_attention_s[b_idx].state_dict()
            
            if self.task_latents:
                state_dict['task_latent_encoder_s'] = self.task_latent_encoder_s[b_idx].state_dict()
        
        if self.deterministic_path:
            state_dict['encoder_d'] = self.encoder_d[b_idx].state_dict()
            state_dict['intra_task_attention_d'] = self.intra_task_attention_d[b_idx].state_dict()
            
            if self.task_latents:
                state_dict['task_latent_encoder_d'] = self.task_latent_encoder_d[b_idx].state_dict()
                
        if self.local_deterministic_path:
            state_dict['encoder_l'] = self.encoder_l[b_idx].state_dict()
            state_dict['intra_task_attention_l'] = self.intra_task_attention_l[b_idx].state_dict()
            
        state_dict['decoder_head'] = self.decoder_head[b_idx].state_dict()
        state_dict['decoder'] = self.decoder[b_idx].state_dict()
        
        return state_dict
    
    def state_dict_(self):
        state_dict = {}
        
        if self.global_latent:
            state_dict['global'] = self.state_dict_global()
            
        for block in self.block_names:
            state_dict[block] = self.state_dict_block(block)
        
        return state_dict
    
    def load_state_dict_global(self, state_dict):
        if self.global_latent:
            if self.stochastic_path:
                self.inter_task_attention_s.load_state_dict(state_dict['inter_task_attention_s'])
                self.global_latent_encoder_s.load_state_dict(state_dict['global_latent_encoder_s'])
                
                if self.task_embedding:
                    self.task_embedding_s.data = state_dict['task_embedding_s']
                
            if self.deterministic_path:
                self.inter_task_attention_d.load_state_dict(state_dict['inter_task_attention_d'])
                self.global_latent_encoder_d.load_state_dict(state_dict['global_latent_encoder_d'])
                
                if self.task_embedding:
                    self.task_embedding_d.data = state_dict['task_embedding_d']
                
            if self.local_deterministic_path:
                self.inter_task_attention_l.load_state_dict(state_dict['inter_task_attention_l'])
                
                if self.task_embedding:
                    self.task_embedding_l.data = state_dict['task_embedding_l']
    
    def load_state_dict_block(self, state_dict, block):
        b_idx = self.block_names.index(block)
        if self.stochastic_path:
            self.encoder_s[b_idx].load_state_dict(state_dict['encoder_s'])
            self.intra_task_attention_s[b_idx].load_state_dict(state_dict['intra_task_attention_s'])
            
            if self.task_latents:
                self.task_latent_encoder_s[b_idx].load_state_dict(state_dict['task_latent_encoder_s'])
                
        if self.deterministic_path:
            self.encoder_d[b_idx].load_state_dict(state_dict['encoder_d'])
            self.intra_task_attention_d[b_idx].load_state_dict(state_dict['intra_task_attention_d'])
            
            if self.task_latents:
                self.task_latent_encoder_d[b_idx].load_state_dict(state_dict['task_latent_encoder_d'])
                
        if self.local_deterministic_path:
            self.encoder_l[b_idx].load_state_dict(state_dict['encoder_l'])
            self.intra_task_attention_l[b_idx].load_state_dict(state_dict['intra_task_attention_l'])
            
        self.decoder_head[b_idx].load_state_dict(state_dict['decoder_head'])
        self.decoder[b_idx].load_state_dict(state_dict['decoder'])
        
    def load_state_dict_(self, state_dict):
        if self.global_latent:
            self.load_state_dict_global(state_dict['global'])
        for block in self.block_names:
            self.load_state_dict_block(state_dict[block], block)
        
    def group_labels(self, Y):
        Y_blocks = {}
        for block in self.block_names:
            Y_blocks[block] = torch.cat([Y[task] for task in self.task_blocks[block]], -1)
            
        return Y_blocks
        
    def ungroup_labels(self, Y_blocks):
        if Y_blocks is None:
            Y = None
        else:
            Y = {}
            for block in self.block_names:
                cut = 0
                for t_idx, task in enumerate(self.task_blocks[block]):
                    if isinstance(Y_blocks[block], tuple):
                        Y[task] = tuple(Y_blocks[block][i][..., cut:cut+self.dim_ys[task]] for i in range(len(Y_blocks[block])))
                    else:
                        Y[task] = Y_blocks[block][..., cut:cut+self.dim_ys[task]]
                    cut += self.dim_ys[task]
                
        return Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=True, ns_G=1, ns_T=1):
        if self.training:
            assert Y_D is not None or not self.stochastic_path
            
            # group labels
            Y_blocks_C = self.group_labels(Y_C)
            if self.stochastic_path:
                Y_blocks_D = self.group_labels(Y_D)
            
            # per-task encoding paths
            masks_C = []
            S_C_s = {}
            S_D_s = {}
            S_C_d = {}
            S_C_l = {}
            for b_idx, block in enumerate(self.block_names):
                # prepare context
                C = torch.cat((X_C, Y_blocks_C[block]), -1) # (B, m, dim_x + dim_y)
                masks_C.append(C[..., -1].isnan()) # (B, m)
                
                if self.stochastic_path:
                    # prepare target
                    D = torch.cat((X_D, Y_blocks_D[block]), -1) # (B, n, dim_x + dim_y)
                
                    # element-wise encoding
                    S_C_s[block] = masked_forward(self.encoder_s[b_idx], C, masks_C[-1], self.dim_hidden)
                    S_D_s[block] = self.encoder_s[b_idx](D)
                    
                    # intra-task attention
                    S_C_s[block] = self.intra_task_attention_s[b_idx](S_C_s[block], masks_C[-1])
                    S_D_s[block] = self.intra_task_attention_s[b_idx](S_D_s[block])
                
                if self.deterministic_path:
                    # element-wise encoding
                    S_C_d[block] = masked_forward(self.encoder_d[b_idx], C, masks_C[-1], self.dim_hidden)
                    
                    # intra-task attention
                    S_C_d[block] = self.intra_task_attention_d[b_idx](S_C_d[block], masks_C[-1])
                
                if self.local_deterministic_path:
                    # element-wise encoding
                    S_C_l[block] = masked_forward(self.encoder_l[b_idx], C, masks_C[-1], self.dim_hidden)
                    
                    # intra-task attention
                    S_C_l[block] = self.intra_task_attention_l[b_idx](X_D, X_C, S_C_l[block], mask_K=masks_C[-1])
            
            # global latent paths
            q_D_G = q_C_G = None
            if self.global_latent:
                if self.stochastic_path:
                    S_C_s_G = torch.stack([S_C_s[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)
                    S_D_s_G = torch.stack([S_D_s[block] for block in self.block_names], 1)  # (B, n_blocks, n, dim_hidden)
                    
                    if self.task_embedding:
                        S_C_s_G = S_C_s_G + self.task_embedding_s.unsqueeze(0)
                        S_D_s_G = S_D_s_G + self.task_embedding_s.unsqueeze(0)

                    S_C_s_G = self.inter_task_attention_s(S_C_s_G)
                    S_D_s_G = self.inter_task_attention_s(S_D_s_G)

                    q_C_G = self.global_latent_encoder_s(S_C_s_G)
                    q_D_G = self.global_latent_encoder_s(S_D_s_G)
                    
                    z = Normal(*q_D_G).rsample()
                    
                if self.deterministic_path:
                    S_C_d_G = torch.stack([S_C_d[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)
                    
                    if self.task_embedding:
                        S_C_d_G = S_C_d_G + self.task_embedding_d.unsqueeze(0)

                    S_C_d_G = self.inter_task_attention_d(S_C_d_G)

                    r = self.global_latent_encoder_d(S_C_d_G)
            
                # local cross-attention path
                if self.local_deterministic_path:
                    S_C_l_G = torch.stack([S_C_l[block] for block in self.block_names], 1)
                    
                    if self.task_embedding:
                        S_C_l_G = S_C_l_G + self.task_embedding_l.unsqueeze(0).unsqueeze(2)
                        
                    r_L = self.inter_task_attention_l(S_C_l_G)
            
            # per-task latent paths
            p_Y = {}
            q_C_T = {}
            q_D_T = {}
            for b_idx, block in enumerate(self.block_names):
                decoder_input = [self.decoder_head[b_idx](X_D)]
                if self.task_latents:
                    if self.stochastic_path:
                        if self.global_latent:
                            q_C_T[block] = self.task_latent_encoder_s[b_idx](torch.cat((z, S_C_s[block]), -1))
                            q_D_T[block] = self.task_latent_encoder_s[b_idx](torch.cat((z, S_D_s[block]), -1))
                        else:
                            q_C_T[block] = self.task_latent_encoder_s[b_idx](S_C_s[block])
                            q_D_T[block] = self.task_latent_encoder_s[b_idx](S_D_s[block])
                            
                        v = Normal(*q_D_T[block]).rsample()
                        decoder_input.append(v.unsqueeze(1).repeat(1, X_D.size(1), 1))
                    else:
                        q_C_T = q_D_T = None
                            
                    if self.deterministic_path:
                        if self.global_latent:
                            r_T = self.task_latent_encoder_d[b_idx](torch.cat((r, S_C_d[block]), -1))
                        else:
                            r_T = self.task_latent_encoder_d[b_idx](S_C_d[block])
                        decoder_input.append(r_T.unsqueeze(1).repeat(1, X_D.size(1), 1))
                            
                    if self.local_deterministic_path:
                        if self.global_latent:
                            decoder_input.append(r_L[:, b_idx])
                        else:
                            decoder_input.append(S_C_l[block])
                        
                else:
                    q_C_T = q_D_T = None
                    if self.stochastic_path:
                        decoder_input.append(z.unsqueeze(1).repeat(1, X_D.size(1), 1))
                    
                    if self.deterministic_path:
                        decoder_input.append(r.unsqueeze(1).repeat(1, X_D.size(1), 1))
                        
                    if self.local_deterministic_path:
                        decoder_input.append(r_L)
                        
                p_Y[block] = self.decoder[b_idx](torch.cat(decoder_input, -1))
                    
            p_Y = self.ungroup_labels(p_Y)

            return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
        else:
            # group labels
            Y_blocks_C = self.group_labels(Y_C)
            
            # per-task encoding paths
            masks_C = []
            S_C_s = {}
            S_C_d = {}
            S_C_l = {}
            for b_idx, block in enumerate(self.block_names):
                # prepare context
                C = torch.cat((X_C, Y_blocks_C[block]), -1) # (B, m, dim_x + dim_y)
                masks_C.append(C[..., -1].isnan()) # (B, m)
                
                if self.stochastic_path:
                    # element-wise encoding
                    S_C_s[block] = masked_forward(self.encoder_s[b_idx], C, masks_C[-1], self.dim_hidden)
                    
                    # intra-task attention
                    S_C_s[block] = self.intra_task_attention_s[b_idx](S_C_s[block], masks_C[-1])
                
                if self.deterministic_path:
                    # element-wise encoding
                    S_C_d[block] = masked_forward(self.encoder_d[b_idx], C, masks_C[-1], self.dim_hidden)
                    
                    # intra-task attention
                    S_C_d[block] = self.intra_task_attention_d[b_idx](S_C_d[block], masks_C[-1])
                
                if self.local_deterministic_path:
                    # element-wise encoding
                    S_C_l[block] = masked_forward(self.encoder_l[b_idx], C, masks_C[-1], self.dim_hidden)
                    
                    # intra-task attention
                    S_C_l[block] = self.intra_task_attention_l[b_idx](X_D, X_C, S_C_l[block], mask_K=masks_C[-1])
            
            # global latent paths
            if self.global_latent:
                if self.stochastic_path:
                    S_C_s_G = torch.stack([S_C_s[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)

                    S_C_s_G = self.inter_task_attention_s(S_C_s_G)

                    q_C_G = self.global_latent_encoder_s(S_C_s_G)
                    
                if self.deterministic_path:
                    S_C_d_G = torch.stack([S_C_d[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)

                    S_C_d_G = self.inter_task_attention_d(S_C_d_G)

                    r = self.global_latent_encoder_d(S_C_d_G)
            
                # local cross-attention path
                if self.local_deterministic_path:
                    S_C_l = torch.stack([S_C_l[block] for block in self.block_names], 1)
                    r_L = self.inter_task_attention_l(S_C_l)
            
            # decoding samples
            if MAP or not self.stochastic_path:
                ns_G = ns_T = 1
            elif not self.global_latent:
                ns_G = 1
            elif not self.task_latents:
                ns_T = 1
            ns = ns_G*ns_T
                
            if self.global_latent and self.stochastic_path:
                if MAP:
                    z = q_C_G[0].unsqueeze(1)
                else:
                    z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)
            
            # per-task latent paths
            p_Y = {}
            for b_idx, block in enumerate(self.block_names):
                decoder_input = [self.decoder_head[b_idx](X_D).unsqueeze(1).repeat(1, ns, 1, 1)]
                if self.task_latents:
                    if self.stochastic_path:
                        if self.global_latent:
                            q_C_T = self.task_latent_encoder_s[b_idx](torch.cat((z, S_C_s[block].unsqueeze(1).repeat(1, ns_G, 1)), -1))
                        else:
                            q_C_T = self.task_latent_encoder_s[b_idx](S_C_s[block].unsqueeze(1))
                        
                        if MAP:
                            v = q_C_T[0]
                        else:
                            v = Normal(*q_C_T).sample((ns_T,)).transpose(1, 2)
                            v = v.reshape(ns, *v.size()[2:]).transpose(0, 1)
                        decoder_input.append(v.unsqueeze(2).repeat(1, 1, X_D.size(1), 1))
                            
                    if self.deterministic_path:
                        if self.global_latent:
                            r_T = self.task_latent_encoder_d[b_idx](torch.cat((r, S_C_d[block]), -1))
                        else:
                            r_T = self.task_latent_encoder_d[b_idx](S_C_d[block])
                        decoder_input.append(r_T.unsqueeze(1).unsqueeze(1).repeat(1, ns, X_D.size(1), 1))
                            
                    if self.local_deterministic_path:
                        if self.global_latent:
                            decoder_input.append(r_L[:, b_idx].unsqueeze(1).repeat(1, ns, 1, 1))
                        else:
                            decoder_input.append(S_C_l[block].unsqueeze(1).repeat(1, ns, 1, 1))
                        
                else:
                    if self.stochastic_path:
                        decoder_input.append(z.unsqueeze(2).repeat(1, 1, X_D.size(1), 1))
                    
                    if self.deterministic_path:
                        decoder_input.append(r.unsqueeze(1).unsqueeze(1).repeat(1, ns, X_D.size(1), 1))
                        
                    if self.local_deterministic_path:
                        decoder_input.append(r_L.unsqueeze(1).repeat(1, ns, 1, 1))
                        
                p_Y[block] = self.decoder[b_idx](torch.cat(decoder_input, -1))
                    
            p_Y = self.ungroup_labels(p_Y)

            return p_Y


class SharedMultiTaskNP(nn.Module):
    def __init__(self, dim_x, dim_ys, dim_hidden, tasks, task_blocks, module_sizes,
                 task_latents, global_latent, stochastic_path, deterministic_path, local_deterministic_path, task_embedding,
                 n_attn_heads, activation, ln, dr, skip, epsilon):

        super().__init__()
        self.tasks = tasks
        for dim_y in dim_ys.values():
            assert dim_y == 1
        self.dim_hidden = dim_hidden
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise NotImplementedError
        
        # configure tasks and task blocks
        self.block_names = []
        self.task_blocks = {}
        for b, task_block in enumerate(task_blocks):
            block = ','.join(task_block)
            self.block_names.append(block)
            self.task_blocks[block] = task_block
            
        # configure paths
        assert stochastic_path or deterministic_path or local_deterministic_path
        self.stochastic_path = stochastic_path
        self.deterministic_path = deterministic_path
        self.local_deterministic_path = local_deterministic_path
        
        # configure LVM
        assert task_latents or global_latent
        self.task_latents = task_latents
        self.global_latent = global_latent
        
        self.task_embedding = task_embedding
        
        # stochastic path
        if self.stochastic_path:
            self.encoder_s = MLP(dim_x + 1, dim_hidden, dim_hidden, module_sizes[0], act_fn, ln, dr, skip)
            
            self.intra_task_attention_s = STEncoder(dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)

            if self.task_latents:
                dim_in = dim_hidden*2 if self.global_latent else dim_hidden
                self.task_latent_encoder_s = LatentMLP(dim_in, dim_hidden, dim_hidden, 2, act_fn, ln, dr, skip, epsilon=epsilon)
            
            if self.global_latent:
                self.inter_task_attention_s = STEncoder(dim_hidden, module_sizes[2], n_attn_heads, act_fn, ln, dr)
                self.global_latent_encoder_s = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2,
                                                         act_fn, ln, dr, skip, epsilon=epsilon)
                
                # task embedding
                if self.task_embedding:
                    self.task_embedding_s = nn.Parameter(torch.randn(len(self.block_names), dim_hidden),
                                                         requires_grad=True)
            
        # deterministic path        
        if self.deterministic_path:
            self.encoder_d = MLP(dim_x + 1, dim_hidden, dim_hidden, module_sizes[0], act_fn, ln, dr, skip)
            
            self.intra_task_attention_d = STEncoder(dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)

            if self.task_latents:
                dim_in = dim_hidden*2 if self.global_latent else dim_hidden
                self.task_latent_encoder_d = LatentMLP(dim_in, dim_hidden, dim_hidden, 2, act_fn, ln, dr, skip, sigma=False)
            
            if self.global_latent:
                self.inter_task_attention_d = STEncoder(dim_hidden, module_sizes[2], n_attn_heads, act_fn, ln, dr)
                self.global_latent_encoder_d = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2,
                                                         act_fn, ln, dr, skip, sigma=False)
                
                # task embedding
                if self.task_embedding:
                    self.task_embedding_d = nn.Parameter(torch.randn(len(self.block_names), dim_hidden),
                                                         requires_grad=True)
        
        # local deterministic path
        if self.local_deterministic_path:
            self.encoder_l = MLP(dim_x + 1, dim_hidden, dim_hidden, module_sizes[0], act_fn, ln, dr, skip)
            
            self.intra_task_attention_l = CrossAttention(dim_x, dim_x, dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)

            if self.global_latent:
                self.inter_task_attention_l = MultiTaskAttention(dim_hidden, module_sizes[2], n_attn_heads,
                                                                 act_fn, ln, dr, pool=(not self.task_latents))
                
                # task embedding
                if self.task_embedding:
                    self.task_embedding_l = nn.Parameter(torch.randn(len(self.block_names), dim_hidden),
                                                         requires_grad=True)
                
        
        # decoder
        self.decoder_head = nn.Linear(dim_x, dim_hidden)
        
        dim_in = dim_hidden * (1 + int(self.stochastic_path) + int(self.deterministic_path) + int(self.local_deterministic_path))
        self.decoder = LatentMLP(dim_in, 1, dim_hidden, module_sizes[3], act_fn, ln, dr, epsilon=epsilon, sigma_act=F.softplus)
    
    def state_dict_(self):
        return self.state_dict()
        
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)
        
    def ungroup_labels(self, Y_blocks):
        if Y_blocks is None:
            Y = None
        elif isinstance(Y_blocks, tuple):
            Y = {}
            for b_idx, block in enumerate(self.block_names):
                if self.training:
                    Y[block] = tuple(Y_block[:, b_idx] for Y_block in Y_blocks)
                else:
                    Y[block] = tuple(Y_block[:, :, b_idx] for Y_block in Y_blocks)
        else:
            if self.training:
                Y = {block: Y_blocks[:, b_idx] for b_idx, block in enumerate(self.block_names)}
            else:
                Y = {block: Y_blocks[:, :, b_idx] for b_idx, block in enumerate(self.block_names)}
                
        return Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=True, ns_G=1, ns_T=1):
        if self.training:
            assert Y_D is not None or not self.stochastic_path
            
            # group labels
            Y_blocks_C = Y_C
            if self.stochastic_path:
                Y_blocks_D = Y_D
            
            B = X_C.size(0)
            T = len(self.block_names)
            C = torch.stack([torch.cat((X_C, Y_blocks_C[block]), -1)
                             for block in self.block_names], 1) # (T, B, m, dim_x + dim_y)
            masks_C = torch.stack([Y_blocks_C[block][..., -1].isnan()
                                   for block in self.block_names], 1) # (T, B, m)
            if self.stochastic_path:
                D = torch.stack([torch.cat((X_D, Y_blocks_D[block]), -1)
                                 for block in self.block_names], 1) # (T, B, n, dim_x + dim_y)
                
                S_C_s = masked_forward(self.encoder_s, C, masks_C, self.dim_hidden)
                S_D_s = self.encoder_s(D)
                
                S_C_s = self.intra_task_attention_s(S_C_s.reshape(B*T, *S_C_s.size()[2:]),
                                                    masks_C.reshape(B*T, *masks_C.size()[2:]))
                S_C_s = S_C_s.reshape(B, T, *S_C_s.size()[1:])
                
                S_D_s = self.intra_task_attention_s(S_D_s.reshape(B*T, *S_D_s.size()[2:]))
                S_D_s = S_D_s.reshape(B, T, *S_D_s.size()[1:])
                
            if self.deterministic_path:
                S_C_d = masked_forward(self.encoder_d, C, masks_C, self.dim_hidden)
                
                S_C_d = self.intra_task_attention_d(S_C_d.reshape(B*T, *S_C_d.size()[2:]),
                                                    masks_C.reshape(B*T, *masks_C.size()[2:]))
                S_C_d = S_C_d.reshape(B, T, *S_C_d.size()[1:])
                
            if self.local_deterministic_path:
                S_C_l = masked_forward(self.encoder_l, C, masks_C, self.dim_hidden)
                
                S_C_l = self.intra_task_attention_l(X_D.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_D.size()[1:]),
                                                    X_C.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_C.size()[1:]),
                                                    S_C_l.reshape(B*T, *S_C_l.size()[2:]),
                                                    mask_K=masks_C.reshape(B*T, *masks_C.size()[2:]))
                S_C_l = S_C_l.reshape(B, T, *S_C_l.size()[1:])
                
            
            # global latent paths
            q_D_G = q_C_G = None
            if self.global_latent:
                if self.stochastic_path:
                    S_C_s_G = S_C_s
                    S_D_s_G = S_D_s
                    
                    if self.task_embedding:
                        S_C_s_G = S_C_s_G + self.task_embedding_s.unsqueeze(0)
                        S_D_s_G = S_D_s_G + self.task_embedding_s.unsqueeze(0)

                    S_C_s_G = self.inter_task_attention_s(S_C_s_G)
                    S_D_s_G = self.inter_task_attention_s(S_D_s_G)

                    q_C_G = self.global_latent_encoder_s(S_C_s_G)
                    q_D_G = self.global_latent_encoder_s(S_D_s_G)
                    
                    z = Normal(*q_D_G).rsample()
                    
                if self.deterministic_path:
                    S_C_d_G = S_C_d
                    
                    if self.task_embedding:
                        S_C_d_G = S_C_d_G + self.task_embedding_d.unsqueeze(0)

                    S_C_d_G = self.inter_task_attention_d(S_C_d_G)

                    r = self.global_latent_encoder_d(S_C_d_G)
            
                # local cross-attention path
                if self.local_deterministic_path:
                    S_C_l_G = S_C_l
                    
                    if self.task_embedding:
                        S_C_l_G = S_C_l_G + self.task_embedding_l.unsqueeze(0).unsqueeze(2)
                        
                    r_L = self.inter_task_attention_l(S_C_l_G)
            
            # per-task latent paths
            decoder_input = [self.decoder_head(X_D).unsqueeze(1).repeat(1, T, 1, 1)]
            
            if self.task_latents:
                if self.stochastic_path:
                    if self.global_latent:
                        q_C_T = self.task_latent_encoder_s(torch.cat((z.unsqueeze(1).repeat(1, T, 1), S_C_s), -1))
                        q_D_T = self.task_latent_encoder_s(torch.cat((z.unsqueeze(1).repeat(1, T, 1), S_D_s), -1))
                    else:
                        q_C_T = self.task_latent_encoder_s(S_C_s)
                        q_D_T = self.task_latent_encoder_s(S_D_s)

                    v = Normal(*q_D_T).rsample()
                    decoder_input.append(v.unsqueeze(2).repeat(1, 1, X_D.size(1), 1))
                else:
                    q_C_T = q_D_T = None

                if self.deterministic_path:
                    if self.global_latent:
                        r_T = self.task_latent_encoder_d(torch.cat((r.unsqueeze(1).repeat(1, T, 1), S_C_d), -1))
                    else:
                        r_T = self.task_latent_encoder_d(S_C_d)
                    decoder_input.append(r_T.unsqueeze(2).repeat(1, 1, X_D.size(1), 1))

                if self.local_deterministic_path:
                    if self.global_latent:
                        decoder_input.append(r_L)
                    else:
                        decoder_input.append(S_C_l)

            else:
                q_C_T = q_D_T = None
                if self.stochastic_path:
                    decoder_input.append(z.unsqueeze(1).unsqueeze(1).repeat(1, T, X_D.size(1), 1))

                if self.deterministic_path:
                    decoder_input.append(r.unsqueeze(1).unsqueeze(1).repeat(1, T, X_D.size(1), 1))

                if self.local_deterministic_path:
                    decoder_input.append(r_L)

            p_Y = self.decoder(torch.cat(decoder_input, -1))
            
            p_Y = self.ungroup_labels(p_Y)
            if q_D_T is not None:
                q_D_T = self.ungroup_labels(q_D_T)
                q_C_T = self.ungroup_labels(q_C_T)

            return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
        else:
            # group labels
            Y_blocks_C = Y_C
            
            B = X_C.size(0)
            T = len(self.block_names)
            C = torch.stack([torch.cat((X_C, Y_blocks_C[block]), -1)
                             for block in self.block_names], 1) # (T, B, m, dim_x + dim_y)
            masks_C = torch.stack([Y_blocks_C[block][..., -1].isnan()
                                   for block in self.block_names], 1) # (T, B, m)
            if self.stochastic_path:
                S_C_s = masked_forward(self.encoder_s, C, masks_C, self.dim_hidden)
                
                S_C_s = self.intra_task_attention_s(S_C_s.reshape(B*T, *S_C_s.size()[2:]),
                                                    masks_C.reshape(B*T, *masks_C.size()[2:]))
                S_C_s = S_C_s.reshape(B, T, *S_C_s.size()[1:])
                
            if self.deterministic_path:
                S_C_d = masked_forward(self.encoder_d, C, masks_C, self.dim_hidden)
                
                S_C_d = self.intra_task_attention_d(S_C_d.reshape(B*T, *S_C_d.size()[2:]),
                                                    masks_C.reshape(B*T, *masks_C.size()[2:]))
                S_C_d = S_C_d.reshape(B, T, *S_C_d.size()[1:])
                
            if self.local_deterministic_path:
                S_C_l = masked_forward(self.encoder_l, C, masks_C, self.dim_hidden)
                
                S_C_l = self.intra_task_attention_l(X_D.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_D.size()[1:]),
                                                    X_C.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_C.size()[1:]),
                                                    S_C_l.reshape(B*T, *S_C_l.size()[2:]),
                                                    mask_K=masks_C.reshape(B*T, *masks_C.size()[2:]))
                S_C_l = S_C_l.reshape(B, T, *S_C_l.size()[1:])
                
            
            # global latent paths
            q_D_G = q_C_G = None
            if self.global_latent:
                if self.stochastic_path:
                    S_C_s_G = S_C_s
                    
                    if self.task_embedding:
                        S_C_s_G = S_C_s_G + self.task_embedding_s.unsqueeze(0)

                    S_C_s_G = self.inter_task_attention_s(S_C_s_G)

                    q_C_G = self.global_latent_encoder_s(S_C_s_G)
                    
                    z = Normal(*q_C_G).sample()
                    
                if self.deterministic_path:
                    S_C_d_G = S_C_d
                    
                    if self.task_embedding:
                        S_C_d_G = S_C_d_G + self.task_embedding_d.unsqueeze(0)

                    S_C_d_G = self.inter_task_attention_d(S_C_d_G)

                    r = self.global_latent_encoder_d(S_C_d_G)
            
                # local cross-attention path
                if self.local_deterministic_path:
                    S_C_l_G = S_C_l
                    
                    if self.task_embedding:
                        S_C_l_G = S_C_l_G + self.task_embedding_l.unsqueeze(0).unsqueeze(2)
                        
                    r_L = self.inter_task_attention_l(S_C_l_G)
            
            # decoding samples
            if MAP or not self.stochastic_path:
                ns_G = ns_T = 1
            elif not self.global_latent:
                ns_G = 1
            elif not self.task_latents:
                ns_T = 1
            ns = ns_G*ns_T
                
            if self.global_latent and self.stochastic_path:
                if MAP:
                    z = q_C_G[0].unsqueeze(1)
                else:
                    z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)
            
            # per-task latent paths
            p_Y = {}
            decoder_input = [self.decoder_head(X_D).unsqueeze(1).unsqueeze(1).repeat(1, ns, T, 1, 1)]
            if self.task_latents:
                if self.stochastic_path:
                    if self.global_latent:
                        q_C_T = self.task_latent_encoder_s(torch.cat((z.unsqueeze(2).repeat(1, 1, T, 1),
                                                                      S_C_s.unsqueeze(1).repeat(1, ns_G, 1, 1)), -1))
                    else:
                        q_C_T = self.task_latent_encoder_s(S_C_s.unsqueeze(1))

                    if MAP:
                        v = q_C_T[0]
                    else:
                        v = Normal(*q_C_T).sample((ns_T,)).transpose(1, 2)
                        v = v.reshape(ns, *v.size()[2:]).transpose(0, 1)
                    decoder_input.append(v.unsqueeze(3).repeat(1, 1, 1, X_D.size(1), 1))

                if self.deterministic_path:
                    if self.global_latent:
                        r_T = self.task_latent_encoder_d(torch.cat((r.unsqueeze(1).repeat(1, T, 1), S_C_d), -1))
                    else:
                        r_T = self.task_latent_encoder_d(S_C_d)
                    decoder_input.append(r_T.unsqueeze(1).unsqueeze(3).repeat(1, ns, 1, X_D.size(1), 1))

                if self.local_deterministic_path:
                    if self.global_latent:
                        decoder_input.append(r_L.unsqueeze(1).repeat(1, ns, 1, 1, 1))
                    else:
                        decoder_input.append(S_C_l.unsqueeze(1).repeat(1, ns, 1, 1, 1))

            else:
                if self.stochastic_path:
                    decoder_input.append(z.unsqueeze(3).repeat(1, 1, 1, X_D.size(1), 1))

                if self.deterministic_path:
                    decoder_input.append(r.unsqueeze(1).unsqueeze(3).repeat(1, ns, 1, X_D.size(1), 1))

                if self.local_deterministic_path:
                    decoder_input.append(r_L.unsqueeze(1).repeat(1, ns, 1, 1, 1))

            p_Y = self.decoder(torch.cat(decoder_input, -1))
                    
            p_Y = self.ungroup_labels(p_Y)

            return p_Y
        
#     def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=True, ns_G=1, ns_T=1):
#         if self.training:
#             assert Y_D is not None or not self.stochastic_path
            
#             # group labels
#             Y_blocks_C = Y_C
#             if self.stochastic_path:
#                 Y_blocks_D = Y_D
            
#             # per-task encoding paths
#             masks_C = []
#             S_C_s = {}
#             S_D_s = {}
#             S_C_d = {}
#             S_C_l = {}
#             for b_idx, block in enumerate(self.block_names):
#                 # prepare context
#                 C = torch.cat((X_C, Y_blocks_C[block]), -1) # (B, m, dim_x + dim_y)
#                 masks_C.append(C[..., -1].isnan()) # (B, m)
                
#                 if self.stochastic_path:
#                     # prepare target
#                     D = torch.cat((X_D, Y_blocks_D[block]), -1) # (B, n, dim_x + dim_y)
                
#                     # element-wise encoding
#                     S_C_s[block] = masked_forward(self.encoder_s, C, masks_C[-1], self.dim_hidden)
#                     S_D_s[block] = self.encoder_s(D)
                    
#                     # intra-task attention
#                     S_C_s[block] = self.intra_task_attention_s(S_C_s[block], masks_C[-1])
#                     S_D_s[block] = self.intra_task_attention_s(S_D_s[block])
                
#                 if self.deterministic_path:
#                     # element-wise encoding
#                     S_C_d[block] = masked_forward(self.encoder_d, C, masks_C[-1], self.dim_hidden)
                    
#                     # intra-task attention
#                     S_C_d[block] = self.intra_task_attention_d(S_C_d[block], masks_C[-1])
                
#                 if self.local_deterministic_path:
#                     # element-wise encoding
#                     S_C_l[block] = masked_forward(self.encoder_l, C, masks_C[-1], self.dim_hidden)
                    
#                     # intra-task attention
#                     S_C_l[block] = self.intra_task_attention_l(X_D, X_C, S_C_l[block], mask_K=masks_C[-1])
            
#             # global latent paths
#             q_D_G = q_C_G = None
#             if self.global_latent:
#                 if self.stochastic_path:
#                     S_C_s_G = torch.stack([S_C_s[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)
#                     S_D_s_G = torch.stack([S_D_s[block] for block in self.block_names], 1)  # (B, n_blocks, n, dim_hidden)
                    
#                     if self.task_embedding:
#                         S_C_s_G = S_C_s_G + self.task_embedding_s.unsqueeze(0)
#                         S_D_s_G = S_D_s_G + self.task_embedding_s.unsqueeze(0)

#                     S_C_s_G = self.inter_task_attention_s(S_C_s_G)
#                     S_D_s_G = self.inter_task_attention_s(S_D_s_G)

#                     q_C_G = self.global_latent_encoder_s(S_C_s_G)
#                     q_D_G = self.global_latent_encoder_s(S_D_s_G)
                    
#                     z = Normal(*q_D_G).rsample()
                    
#                 if self.deterministic_path:
#                     S_C_d_G = torch.stack([S_C_d[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)
                    
#                     if self.task_embedding:
#                         S_C_d_G = S_C_d_G + self.task_embedding_d.unsqueeze(0)

#                     S_C_d_G = self.inter_task_attention_d(S_C_d_G)

#                     r = self.global_latent_encoder_d(S_C_d_G)
            
#                 # local cross-attention path
#                 if self.local_deterministic_path:
#                     S_C_l_G = torch.stack([S_C_l[block] for block in self.block_names], 1)
                    
#                     if self.task_embedding:
#                         S_C_l_G = S_C_l_G + self.task_embedding_l.unsqueeze(0).unsqueeze(2)
                        
#                     r_L = self.inter_task_attention_l(S_C_l_G)
            
#             # per-task latent paths
#             p_Y = {}
#             q_C_T = {}
#             q_D_T = {}
#             X_D_proj = self.decoder_head(X_D)
#             for b_idx, block in enumerate(self.block_names):
#                 decoder_input = [X_D_proj]
#                 if self.task_latents:
#                     if self.stochastic_path:
#                         if self.global_latent:
#                             q_C_T[block] = self.task_latent_encoder_s(torch.cat((z, S_C_s[block]), -1))
#                             q_D_T[block] = self.task_latent_encoder_s(torch.cat((z, S_D_s[block]), -1))
#                         else:
#                             q_C_T[block] = self.task_latent_encoder_s(S_C_s[block])
#                             q_D_T[block] = self.task_latent_encoder_s(S_D_s[block])
                            
#                         v = Normal(*q_D_T[block]).rsample()
#                         decoder_input.append(v.unsqueeze(1).repeat(1, X_D.size(1), 1))
#                     else:
#                         q_C_T = q_D_T = None
                            
#                     if self.deterministic_path:
#                         if self.global_latent:
#                             r_T = self.task_latent_encoder_d(torch.cat((r, S_C_d[block]), -1))
#                         else:
#                             r_T = self.task_latent_encoder_d(S_C_d[block])
#                         decoder_input.append(r_T.unsqueeze(1).repeat(1, X_D.size(1), 1))
                            
#                     if self.local_deterministic_path:
#                         if self.global_latent:
#                             decoder_input.append(r_L[:, b_idx])
#                         else:
#                             decoder_input.append(S_C_l[block])
                        
#                 else:
#                     q_C_T = q_D_T = None
#                     if self.stochastic_path:
#                         decoder_input.append(z.unsqueeze(1).repeat(1, X_D.size(1), 1))
                    
#                     if self.deterministic_path:
#                         decoder_input.append(r.unsqueeze(1).repeat(1, X_D.size(1), 1))
                        
#                     if self.local_deterministic_path:
#                         decoder_input.append(r_L)
                        
#                 p_Y[block] = self.decoder(torch.cat(decoder_input, -1))
                    
#             return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
#         else:
#             # group labels
#             Y_blocks_C = Y_C
            
#             # per-task encoding paths
#             masks_C = []
#             S_C_s = {}
#             S_C_d = {}
#             S_C_l = {}
#             for b_idx, block in enumerate(self.block_names):
#                 # prepare context
#                 C = torch.cat((X_C, Y_blocks_C[block]), -1) # (B, m, dim_x + dim_y)
#                 masks_C.append(C[..., -1].isnan()) # (B, m)
                
#                 if self.stochastic_path:
#                     # element-wise encoding
#                     S_C_s[block] = masked_forward(self.encoder_s, C, masks_C[-1], self.dim_hidden)
                    
#                     # intra-task attention
#                     S_C_s[block] = self.intra_task_attention_s(S_C_s[block], masks_C[-1])
                
#                 if self.deterministic_path:
#                     # element-wise encoding
#                     S_C_d[block] = masked_forward(self.encoder_d, C, masks_C[-1], self.dim_hidden)
                    
#                     # intra-task attention
#                     S_C_d[block] = self.intra_task_attention_d(S_C_d[block], masks_C[-1])
                
#                 if self.local_deterministic_path:
#                     # element-wise encoding
#                     S_C_l[block] = masked_forward(self.encoder_l, C, masks_C[-1], self.dim_hidden)
                    
#                     # intra-task attention
#                     S_C_l[block] = self.intra_task_attention_l(X_D, X_C, S_C_l[block], mask_K=masks_C[-1])
            
#             # global latent paths
#             if self.global_latent:
#                 if self.stochastic_path:
#                     S_C_s_G = torch.stack([S_C_s[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)

#                     S_C_s_G = self.inter_task_attention_s(S_C_s_G)

#                     q_C_G = self.global_latent_encoder_s(S_C_s_G)
                    
#                 if self.deterministic_path:
#                     S_C_d_G = torch.stack([S_C_d[block] for block in self.block_names], 1)  # (B, n_blocks, m, dim_hidden)

#                     S_C_d_G = self.inter_task_attention_d(S_C_d_G)

#                     r = self.global_latent_encoder_d(S_C_d_G)
            
#                 # local cross-attention path
#                 if self.local_deterministic_path:
#                     S_C_l = torch.stack([S_C_l[block] for block in self.block_names], 1)
#                     r_L = self.inter_task_attention_l(S_C_l)
            
#             # decoding samples
#             if MAP or not self.stochastic_path:
#                 ns_G = ns_T = 1
#             elif not self.global_latent:
#                 ns_G = 1
#             elif not self.task_latents:
#                 ns_T = 1
#             ns = ns_G*ns_T
                
#             if self.global_latent and self.stochastic_path:
#                 if MAP:
#                     z = q_C_G[0].unsqueeze(1)
#                 else:
#                     z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)
            
#             # per-task latent paths
#             p_Y = {}
#             X_D_proj = self.decoder_head(X_D)
#             for b_idx, block in enumerate(self.block_names):
#                 decoder_input = [X_D_proj.unsqueeze(1).repeat(1, ns, 1, 1)]
#                 if self.task_latents:
#                     if self.stochastic_path:
#                         if self.global_latent:
#                             q_C_T = self.task_latent_encoder_s(torch.cat((z, S_C_s[block].unsqueeze(1).repeat(1, ns_G, 1)), -1))
#                         else:
#                             q_C_T = self.task_latent_encoder_s(S_C_s[block].unsqueeze(1))
                        
#                         if MAP:
#                             v = q_C_T[0]
#                         else:
#                             v = Normal(*q_C_T).sample((ns_T,)).transpose(1, 2)
#                             v = v.reshape(ns, *v.size()[2:]).transpose(0, 1)
#                         decoder_input.append(v.unsqueeze(2).repeat(1, 1, X_D.size(1), 1))
                            
#                     if self.deterministic_path:
#                         if self.global_latent:
#                             r_T = self.task_latent_encoder_d(torch.cat((r, S_C_d[block]), -1))
#                         else:
#                             r_T = self.task_latent_encoder_d(S_C_d[block])
#                         decoder_input.append(r_T.unsqueeze(1).unsqueeze(1).repeat(1, ns, X_D.size(1), 1))
                            
#                     if self.local_deterministic_path:
#                         if self.global_latent:
#                             decoder_input.append(r_L[:, b_idx].unsqueeze(1).repeat(1, ns, 1, 1))
#                         else:
#                             decoder_input.append(S_C_l[block].unsqueeze(1).repeat(1, ns, 1, 1))
                        
#                 else:
#                     if self.stochastic_path:
#                         decoder_input.append(z.unsqueeze(2).repeat(1, 1, X_D.size(1), 1))
                    
#                     if self.deterministic_path:
#                         decoder_input.append(r.unsqueeze(1).unsqueeze(1).repeat(1, ns, X_D.size(1), 1))
                        
#                     if self.local_deterministic_path:
#                         decoder_input.append(r_L.unsqueeze(1).repeat(1, ns, 1, 1))
                        
#                 p_Y[block] = self.decoder(torch.cat(decoder_input, -1))
                    
#             return p_Y
