from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
from .attention import SAB, PMA


__all__ = ['EfficientWnet', 'get_model']


# def get_blocks_to_be_concat(model, x):
#     shapes = set()
#     blocks = OrderedDict()
#     hooks = []
#     count = 0

#     def register_hook(module):

#         def hook(module, input, output):
#             try:
#                 nonlocal count
#                 if module.name == f'blocks_{count}_output_batch_norm':
#                     count += 1
#                     shape = output.size()[-2:]
#                     if shape not in shapes:
#                         shapes.add(shape)
#                         blocks[module.name] = output

#                 elif module.name == 'head_swish':
#                     blocks.popitem()
#                     blocks[module.name] = output

#             except AttributeError:
#                 pass

#         if (
#                 not isinstance(module, nn.Sequential)
#                 and not isinstance(module, nn.ModuleList)
#                 and not (module == model)
#         ):
#             hooks.append(module.register_forward_hook(hook))

#     # register hook
#     model.apply(register_hook)

#     # make a forward pass to trigger the hooks
#     model(x)

#     # remove these hooks
#     for h in hooks:
#         h.remove()

#     return blocks


class EfficientWnet(nn.Module):
    def __init__(self, encoder_x, encoder_xy, out_channels=1, enc_attn=False, dec_attn=False, n_attn_layers=1):
        super().__init__()

        self.encoder_x = encoder_x
        self.encoder_xy = encoder_xy
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn

        self.conditioners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.size[3-i], self.size[3-i]),
                nn.GELU(),
                nn.Linear(self.size[3-i], self.size[3-i]),
                nn.GELU(),
                nn.Linear(self.size[3-i], self.size[3-i])
            )
            for i in range(4)
        ] + [
            nn.Sequential(
                nn.Linear(self.n_channels, self.n_channels),
                nn.GELU(),
                nn.Linear(self.n_channels, self.n_channels),
                nn.GELU(),
                nn.Linear(self.n_channels, self.n_channels)
            )
        ])
        
        if self.enc_attn:
            self.inter_channel_attns_enc = nn.ModuleList([nn.Sequential(*[SAB(self.size[3-i]) for _ in range(n_attn_layers)]) for i in range(4)] + \
                                                         [nn.Sequential(*[SAB(self.n_channels) for _ in range(n_attn_layers)])])
        if self.dec_attn:
            self.inter_channel_attns_dec = nn.ModuleList([nn.Sequential(*[SAB(512//(2**i)) for _ in range(n_attn_layers)]) for i in range(4)])
        
        self.pre_conv1 = AdaIN(self.n_channels, self.n_channels, self.n_channels)
        self.up_conv1 = UpConv(self.n_channels, 512)
        
        self.pre_convs = nn.ModuleList([AdaIN(self.size[i], self.size[i], self.size[i]) for i in range(4)])
        self.post_convs = nn.ModuleList([AdaIN(512//(2**i) + self.size[i], 512//(2**i), self.size[i]) for i in range(4)])
        self.up_convs = nn.ModuleList([UpConv(512//(2**i), 512//(2**(i+1))) for i in range(4)])

        self.final_conv = Conv(32, out_channels, kernel_size=1)
        
    def domain_parameters(self):
        domain_modules = [self.encoder_x]
        for domain_module in domain_modules:
            for param in domain_module.parameters():
                yield param
        
    def task_parameters(self):
        task_modules = [self.encoder_xy, self.conditioners, self.pre_conv1, self.up_conv1,
                        self.pre_convs, self.post_convs, self.up_convs, self.final_conv]
        if self.enc_attn:
            task_modules.append(self.inter_channel_attns_enc)
        if self.dec_attn:
            task_modules.append(self.inter_channel_attns_dec)
            
        for task_module in task_modules:
            for param in task_module.parameters():
                yield param
        

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder_x.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [80, 40, 24, 16],
                     'efficientnet-b1': [80, 40, 24, 16],
                     'efficientnet-b2': [88, 48, 24, 16],
                     'efficientnet-b3': [96, 48, 32, 24],
                     'efficientnet-b4': [112, 56, 32, 24],
                     'efficientnet-b5': [128, 64, 40, 24],
                     'efficientnet-b6': [144, 72, 40, 32],
                     'efficientnet-b7': [160, 80, 48, 32]}
        return size_dict[self.encoder_x.name]
    
    def forward(self, X_C, Y_C, X_D):
        B, N = X_D.size()[:2]
        O = Y_C.size(2)
        
        blocks_context = self.encode_context(X_C, Y_C)
        blocks_target = self.encode_target(X_D)
        
        output = self.decode(blocks_context, blocks_target)
        output = output.reshape(B, N, O, *X_D.size()[3:])
        
        return output
    
    def encode_context(self, X_C, Y_C):
        # reshape data
        B, N = X_C.size()[:2]
        O = Y_C.size(2)
        x = X_C.view(B*N, *X_C.size()[2:])
        y = Y_C.view(B*N, *Y_C.size()[2:])
        
        # encode context by U-Net
        blocks_context = OrderedDict()
        for channel in range(y.size(1)):
            xy = torch.cat((x, y[:, channel].unsqueeze(1)), 1)
#             blocks = get_blocks_to_be_concat(self.encoder_xy, xy)
            blocks = self.encoder_xy(xy)
            for key in blocks:
                H = blocks[key]
                h = H.view(B, N, *H.size()[1:]).mean(3).mean(3)
                if key in blocks_context:
                    blocks_context[key].append(h)
                else:
                    blocks_context[key] = [h]
        
        # aggregate examples and apply inter-channel interaction
        for i, key in enumerate(blocks_context):
            h = torch.stack(blocks_context[key], 2)
            h = self.conditioners[i](h)
            h = h.mean(1)
                
            if self.enc_attn:
                h = self.inter_channel_attns_enc[i](h)
                
            blocks_context[key] = h
        
        return blocks_context
    
    def encode_target(self, X_D):
        B, N = X_D.size()[:2]
        x = X_D.view(B*N, *X_D.size()[2:])
#         blocks_target = get_blocks_to_be_concat(self.encoder_x, x)
        blocks_target = self.encoder_x(x)
        
        for key in blocks_target:
            blocks_target[key] = blocks_target[key].view(B, N, *blocks_target[key].size()[1:])
        
        return blocks_target

    def decode(self, blocks_context, blocks_target):
        # pop bottom-level representations for context and target input
        _, x = blocks_target.popitem()
        _, h = blocks_context.popitem()
        
        # repeat for multi-channel prediction
        x = x.unsqueeze(2).repeat(1, 1, h.size(1), 1, 1, 1)
        B, N, O = x.size()[:3]
        
        # bottom-level decoding
        x = self.pre_conv1(x, h)
        x = self.up_conv1(x)
        
        for i in range(4):
            _, x_low = blocks_target.popitem()
            _, h_low = blocks_context.popitem()
            x_low = x_low.unsqueeze(2).repeat(1, 1, O, 1, 1, 1)
            x_low = self.pre_convs[i](x_low, h_low)
            x = torch.cat((x, x_low), 3)
            x = self.post_convs[i](x, h_low) # (B, N, O, C, H, W)
            if self.dec_attn:
                H, W = x.size()[4:]
                x = x.permute(0, 1, 4, 5, 2, 3).reshape(B*N*H*W, O, x.size(3))
                x = self.inter_channel_attns_dec[i](x)
                x = x.reshape(B, N, H, W, O, x.size(2)).permute(0, 1, 4, 5, 2, 3)
            x = self.up_convs[i](x)
                
        x = self.final_conv(x)

        return x

def get_model(model_type='efficientnet-b0', enc_attn=False, dec_attn=False, n_attn_layers=1):
    encoder_x = EfficientNet.encoder(model_type, pretrained=False)
    encoder_xy = EfficientNet.encoder(model_type, pretrained=False, in_channels=4)
    model = EfficientWnet(encoder_x, encoder_xy, out_channels=1, enc_attn=enc_attn, dec_attn=dec_attn, n_attn_layers=n_attn_layers)
    return model