import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import ConvEncoder, ConvDecoder, LatentMLP, STEncoder, CrossAttention, MultiTaskAttention


def get_model(config, device):
        return ConvMultiTaskCNP(config.dim_x, config.dim_hidden, config.module_sizes,
                                config.local_deterministic_path, config.n_attn_heads, config.activation,
                                config.layernorm, config.dropout, config.skip).to(device)
    

class DataParallel(nn.DataParallel):
    def state_dict_(self):
        return self.module.state_dict_()
    
    def load_state_dict_(self, state_dict):
        self.module.load_state_dict_(state_dict)
        

class ConvMultiTaskCNP(nn.Module):
    def __init__(self, dim_x, dim_hidden, module_sizes, local_deterministic_path,
                 n_attn_heads, activation, ln, dr, skip):

        super().__init__()
        self.dim_hidden = dim_hidden
        self.local_deterministic_path = local_deterministic_path
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise NotImplementedError
            
        # deterministic path        
        self.encoder_d = ConvEncoder(dim_x + 1, 32, dim_hidden, act_fn, dr)
        self.intra_task_attention_d = STEncoder(dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)

        self.task_latent_encoder_d = LatentMLP(dim_hidden*2, dim_hidden, dim_hidden, 2, act_fn, ln, dr, skip, sigma=False)

        self.inter_task_attention_d = STEncoder(dim_hidden, module_sizes[2], n_attn_heads, act_fn, ln, dr)
        self.global_latent_encoder_d = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2,
                                                 act_fn, ln, dr, skip, sigma=False)
        
        # local deterministic path
        if self.local_deterministic_path:
            self.encoder_l = ConvEncoder(dim_x + 1, 32, dim_hidden, act_fn, dr)
            
            self.intra_task_attention_l = CrossAttention(dim_hidden, dim_hidden, dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr, proj=False)

            self.inter_task_attention_l = MultiTaskAttention(dim_hidden, module_sizes[2], n_attn_heads,
                                                             act_fn, ln, dr, pool=False)
                
        
        # decoder
        self.decoder_head = ConvEncoder(dim_x, 32, dim_hidden, act_fn, dr)
        
        dim_in = dim_hidden * (2 + int(self.local_deterministic_path))
        self.decoder = ConvDecoder(dim_in, dim_hidden, 1, act_fn, dr)
        
    def forward(self, X_C, Y_C, X_D):
        '''
        X_C: (B, N, dim_x, H, W)
        Y_C: (B, N, H, W)
        '''
        B, N = X_C.size()[:2]
        M = X_D.size(1)
        T = len(torch.unique(Y_C))
        Y_C = F.one_hot(Y_C, T).permute(0, 4, 1, 2, 3) # (B, T, N, H, W)

        C = torch.cat((X_C.unsqueeze(1).repeat(1, T, 1, 1, 1, 1),
                       Y_C.unsqueeze(3)), 3) # (B, T, dim_x+1, N, H, W)

        # target input encoding
        X_D_proj = self.decoder_head(X_D.reshape(B*M, *X_D.size()[2:])) #(B*M, dim_hidden)
        X_D_proj = X_D_proj.reshape(B, M, X_D_proj.size(1)) # (B, M, dim_hidden)

        # intra-task encoding
        S_C_d = self.encoder_d(C.reshape(B*T*N, *C.size()[3:])) # (B*T*N, dim_hidden)
        S_C_d = self.intra_task_attention_d(S_C_d.reshape(B*T, N, S_C_d.size(1))) # (B*T, dim_hidden)
        S_C_d = S_C_d.reshape(B, T, S_C_d.size(1)) # (B, T, dim_hidden)

        # inter-task encoding
        S_C_d_G = self.inter_task_attention_d(S_C_d) # (B, dim_hidden)
        r = self.global_latent_encoder_d(S_C_d_G) # (B, dim_hidden)

        # local attention path
        if self.local_deterministic_path:
            # context input encoding
            X_C_proj = self.decoder_head(X_C.reshape(B*N, *X_C.size()[2:])) #(B*N, dim_hidden)
            X_C_proj = X_C_proj.reshape(B, N, X_C_proj.size(1)) # (B, N, dim_hidden)

            # intra-task encoding
            S_C_l = self.encoder_l(C.reshape(B*T*N, *C.size()[3:])) # (B*T*N, dim_hidden)
            S_C_l = self.intra_task_attention_l(X_D_proj.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_D_proj.size()[1:]),
                                                X_C_proj.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_C_proj.size()[1:]),
                                                S_C_l.reshape(B*T, N, S_C_l.size(1))) # (B*T, M, dim_hidden)
            S_C_l = S_C_l.reshape(B, T, *S_C_l.size()[1:]) # (B, T, M, dim_hidden)

            # inter-task encoding
            r_L = self.inter_task_attention_l(S_C_l) # (B, T, M, dim_hidden)

        # per-task decoding
        decoder_input = [X_D_proj.unsqueeze(1).repeat(1, T, 1, 1)]

        r_T = self.task_latent_encoder_d(torch.cat((r.unsqueeze(1).repeat(1, T, 1), S_C_d), -1)) # (B, T, dim_hidden)
        decoder_input.append(r_T.unsqueeze(2).repeat(1, 1, M, 1)) # (B, T, M, dim_hidden)

        if self.local_deterministic_path:
            decoder_input.append(r_L)

        decoder_input = torch.cat(decoder_input, -1) # (B, T, M, len(decoder_input)*dim_hidden)
        decoder_input = decoder_input.reshape(B*T*M, -1) # (B*T*M, len(decoder_input)*dim_hidden)

        Y_D_logits = F.interpolate(self.decoder(decoder_input), X_D.size()[3:], mode='bilinear', align_corners=True) # (B*T*M, 1, H, W)
        Y_D_logits = Y_D_logits.reshape(B, T, M, *Y_D_logits.size()[2:]) # (B, T, M, H, W)

        return Y_D_logits