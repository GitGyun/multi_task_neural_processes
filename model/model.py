import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import ConvEncoder, ConvDecoder, LatentMLP, STEncoder, CrossAttention, MultiTaskAttention


def get_model(config, device):
        return ConvMultiTaskCNP(config.dim_x, config.dim_hidden, config.module_sizes,
                                config.global_path, config.local_path, config.inter_channel_attention,
                                config.n_attn_heads, config.activation,
                                config.layernorm, config.dropout, config.skip).to(device)
    

class DataParallel(nn.DataParallel):
    def state_dict_(self):
        return self.module.state_dict_()
    
    def load_state_dict_(self, state_dict):
        self.module.load_state_dict_(state_dict)
        

class ConvMultiTaskCNP(nn.Module):
    def __init__(self, dim_x, dim_hidden, module_sizes, global_path, local_path, inter_channel_attention,
                 n_attn_heads, activation, ln, dr, skip):

        super().__init__()
        self.dim_hidden = dim_hidden
        assert global_path or local_path
        self.global_path = global_path
        self.local_path = local_path
        self.inter_channel_attention = inter_channel_attention
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise NotImplementedError
            
        # example-level self-attention path
        if self.global_path:
            self.encoder_G = ConvEncoder(dim_x + 1, 32, dim_hidden, act_fn, dr)
            self.inter_example_attention_G = STEncoder(dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr)

            # channel-level self-attention path
            if self.inter_channel_attention:
                self.inter_channel_attention_G = STEncoder(dim_hidden, module_sizes[2], n_attn_heads, act_fn, ln, dr)
                self.example_channel_merger_G = LatentMLP(dim_hidden*2, dim_hidden, dim_hidden, 2,
                                                          act_fn, ln, dr, skip, sigma=False)
        
        # example-level cross-attention path
        if self.local_path:
            self.encoder_L = ConvEncoder(dim_x + 1, 32, dim_hidden, act_fn, dr)
            self.inter_example_attention_L = CrossAttention(dim_hidden, dim_hidden, dim_hidden, module_sizes[1], n_attn_heads, act_fn, ln, dr, proj=False)
            
            # channel-level self-attention path
            if self.inter_channel_attention:
                self.inter_channel_attention_L = MultiTaskAttention(dim_hidden, module_sizes[2], n_attn_heads,
                                                                    act_fn, ln, dr, pool=False)
                
        
        # decoder
        self.decoder_head = ConvEncoder(dim_x, 32, dim_hidden, act_fn, dr)
        
        dim_in = dim_hidden * (1 + int(self.global_path) + int(self.local_path))
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

        # global self-attention path
        if self.global_path:
            # example-level encoding
            C_G = self.encoder_G(C.reshape(B*T*N, *C.size()[3:])) # (B*T*N, dim_hidden)
            R_G = self.inter_example_attention_G(C_G.reshape(B*T, N, C_G.size(1))) # (B*T, dim_hidden)
            R_G = R_G.reshape(B, T, R_G.size(1)) # (B, T, dim_hidden)

            # chennel-level encoding
            if self.inter_channel_attention:
                R_G_pool = self.inter_channel_attention_G(R_G) # (B, dim_hidden)
                R_G = self.example_channel_merger_G(torch.cat((R_G_pool.unsqueeze(1).repeat(1, T, 1),
                                                               R_G), -1)) # (B, T, dim_hidden)

        # local cross-attention path
        if self.local_path:
            # context input encoding
            X_C_proj = self.decoder_head(X_C.reshape(B*N, *X_C.size()[2:])) #(B*N, dim_hidden)
            X_C_proj = X_C_proj.reshape(B, N, X_C_proj.size(1)) # (B, N, dim_hidden)

            # example-level encoding
            C_L = self.encoder_L(C.reshape(B*T*N, *C.size()[3:])) # (B*T*N, dim_hidden)
            R_L = self.inter_example_attention_L(X_D_proj.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_D_proj.size()[1:]),
                                                 X_C_proj.unsqueeze(1).repeat(1, T, 1, 1).reshape(B*T, *X_C_proj.size()[1:]),
                                                 C_L.reshape(B*T, N, C_L.size(1))) # (B*T, M, dim_hidden)
            R_L = R_L.reshape(B, T, *R_L.size()[1:]) # (B, T, M, dim_hidden)

            # inter-task encoding
            if self.inter_channel_attention:
                R_L = self.inter_channel_attention_L(R_L) # (B, T, M, dim_hidden)

        # per-task decoding
        decoder_input = [X_D_proj.unsqueeze(1).repeat(1, T, 1, 1)]
        if self.global_path:
            decoder_input.append(R_G.unsqueeze(2).repeat(1, 1, M, 1)) # (B, T, M, dim_hidden)
        if self.local_path:
            decoder_input.append(R_L)

        decoder_input = torch.cat(decoder_input, -1) # (B, T, M, len(decoder_input)*dim_hidden)
        decoder_input = decoder_input.reshape(B*T*M, -1) # (B*T*M, len(decoder_input)*dim_hidden)

        Y_D_logits = F.interpolate(self.decoder(decoder_input), X_D.size()[3:], mode='bilinear', align_corners=True) # (B*T*M, 1, H, W)
        Y_D_logits = Y_D_logits.reshape(B, T, M, *Y_D_logits.size()[2:]) # (B, T, M, H, W)

        return Y_D_logits