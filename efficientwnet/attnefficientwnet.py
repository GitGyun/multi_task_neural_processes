from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
from .attention import SAB, PMA, CAB


__all__ = ['AttnEfficientWnet', 'get_model2']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class AttnEfficientWnet(nn.Module):
    def __init__(self, encoder_x, encoder_xy, out_channels=1, double_cross=False):
        super().__init__()

        self.encoder_x = encoder_x
        self.encoder_xy = encoder_xy
        self.double_cross = double_cross

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
        
        self.example_attns = nn.ModuleList([CAB(self.n_channels)] + [CAB(self.size[i]) for i in range(4)])
        if self.double_cross:
            self.channel_attns = nn.ModuleList([CAB(self.n_channels)] + [CAB(self.size[i]) for i in range(4)])
        else:
            self.channel_attns = nn.ModuleList([SAB(self.n_channels)] + [SAB(self.size[i]) for i in range(4)])
        self.pixel_convs = nn.ModuleList([Conv(self.n_channels, self.n_channels, kernel_size=3, padding=1)] + \
                                         [Conv(self.size[i], self.size[i], kernel_size=3, padding=1) for i in range(4)])
        
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
                        self.pre_convs, self.post_convs, self.up_convs, self.final_conv,
                        self.example_attns, self.channel_attns, self.pixel_convs]
            
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
        blocks_context_input = self.encode_inputs(X_C)
        blocks_target_input = self.encode_inputs(X_D)
        
        output = self.decode(blocks_target_input, blocks_context_input, blocks_context)
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
            blocks = get_blocks_to_be_concat(self.encoder_xy, xy)
            for key in blocks:
                H = blocks[key]
#                 h = H.view(B, N, *H.size()[1:]).mean(3).mean(3)
                h = H.view(B, N, *H.size()[1:]).permute(0, 1, 3, 4, 2)
                if key in blocks_context:
                    blocks_context[key].append(h)
                else:
                    blocks_context[key] = [h]
        
        # aggregate examples and apply inter-channel interaction
        for i, key in enumerate(blocks_context):
            h = torch.stack(blocks_context[key], 2)
            h = self.conditioners[i](h)
            blocks_context[key] = h
        
        return blocks_context
    
    def encode_inputs(self, X):
        B, N = X.size()[:2]
        x = X.view(B*N, *X.size()[2:])
        blocks_input = get_blocks_to_be_concat(self.encoder_x, x)
        
        for key in blocks_input:
            blocks_input[key] = blocks_input[key].view(B, N, *blocks_input[key].size()[1:])
        
        return blocks_input
    
    def attentive_decoding(self, x_D, x_C, h, level):
        B, M, O, C, H, W = x_D.size()
        N = x_C.size(1)
        
        # cross-attention on examples
        Q = x_D.permute(0, 2, 4, 5, 1, 3).reshape(B*O*H*W, M, C)
        K = x_C.permute(0, 2, 4, 5, 1, 3).reshape(B*O*H*W, N, C)
        V = h.permute(0, 2, 4, 5, 1, 3).reshape(B*O*H*W, N, C)
        h = self.example_attns[level](Q, K, V)
        
        if self.double_cross:
            # cross-attention on channels
            h = h.reshape(B, O, H, W, M, C)
            h = h.permute(0, 4, 2, 3, 1, 5).reshape(B*M*H*W, O, C)
            V = V.reshape(B, O, H, W, N, C)
            V = V.permute(0, 4, 2, 3, 1, 5).reshape(B*N*H*W, O, C)
            h = self.channel_attns[level](h, V)
        else:
            # self-attention on channels
            h = h.reshape(B, O, H, W, M, C)
            h = h.permute(0, 4, 2, 3, 1, 5).reshape(B*M*H*W, O, C)
            h = self.channel_attns[level](h)
        
        # convolution on pixels and aggregation
        h = h.reshape(B, M, H, W, O, C)
        h = h.permute(0, 1, 4, 5, 2, 3).reshape(B*M*O, C, H, W)
        h = self.pixel_convs[level](h)
        h = h.reshape(B, M, O, C, H, W).mean(4).mean(4)
        
        return h

    def decode(self, blocks_target_input, blocks_context_input, blocks_context):
        # pop bottom-level representations for context and target input
        _, x_D = blocks_target_input.popitem() # (B, M, C, H, W)
        _, x_C = blocks_context_input.popitem() # (B, N, C, H, W)
        _, h = blocks_context.popitem() # (B, N, O, C, H, W)
        
        # repeat for multi-channel prediction
        O = h.size(2)
        x_D = x_D.unsqueeze(2).repeat(1, 1, O, 1, 1, 1)
        x_C = x_C.unsqueeze(2).repeat(1, 1, O, 1, 1, 1)
        
        # bottom-level decoding
        h = self.attentive_decoding(x_D, x_C, h, 0)
        x_D = self.pre_conv1(x_D, h)
        x_D = self.up_conv1(x_D)
        
        for i in range(4):
            _, x_D_low = blocks_target_input.popitem()
            _, x_C_low = blocks_context_input.popitem()
            _, h_low = blocks_context.popitem()
            
            x_D_low = x_D_low.unsqueeze(2).repeat(1, 1, O, 1, 1, 1)
            x_C_low = x_C_low.unsqueeze(2).repeat(1, 1, O, 1, 1, 1)
            
            h_low = self.attentive_decoding(x_D_low, x_C_low, h_low, i+1)
            x_D_low = self.pre_convs[i](x_D_low, h_low)
            x_D = torch.cat((x_D, x_D_low), 3)
            x_D = self.post_convs[i](x_D, h_low) # (B, N, O, C, H, W)
            x_D = self.up_convs[i](x_D)
                
        x_D = self.final_conv(x_D)

        return x_D

def get_model2(model_type='efficientnet-b0', double_cross=False):
    encoder_x = EfficientNet.encoder(model_type, pretrained=False)
    encoder_xy = EfficientNet.encoder(model_type, pretrained=False, in_channels=4)
    model = AttnEfficientWnet(encoder_x, encoder_xy, out_channels=1, double_cross=double_cross)
    return model