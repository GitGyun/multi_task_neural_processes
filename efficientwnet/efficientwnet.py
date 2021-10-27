from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet
from .attention import SAB, PMA


__all__ = ['EfficientWnet', 'get_model']



class EfficientWnet(nn.Module):
    def __init__(self, context_encoder, target_encoder):
        super().__init__()

        self.context_encoder = context_encoder
        self.target_encoder = target_encoder

        self.pre_conv1 = AdaIN(self.n_channels, self.n_channels, self.n_channels)
        self.up_conv1 = UpConv(self.n_channels, 512)
        
        self.pre_convs = nn.ModuleList([AdaIN(self.size[i], self.size[i], self.size[i]) for i in range(4)])
        self.post_convs = nn.ModuleList([AdaIN(512//(2**i) + self.size[i], 512//(2**i), self.size[i]) for i in range(4)])
        self.up_convs = nn.ModuleList([UpConv(512//(2**i), 512//(2**(i+1))) for i in range(4)])

        self.final_conv = Conv(32, 1, kernel_size=1)
        
    def domain_parameters(self):
        domain_modules = [self.target_encoder]
        for domain_module in domain_modules:
            for param in domain_module.parameters():
                yield param
        
    def task_parameters(self):
        task_modules = [self.context_encoder, self.pre_conv1, self.up_conv1,
                        self.pre_convs, self.post_convs, self.up_convs, self.final_conv]
            
        for task_module in task_modules:
            for param in task_module.parameters():
                yield param

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.target_encoder.name]

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
        return size_dict[self.target_encoder.name]
    
    def forward(self, X_C, Y_C, X_D):
        B, N = X_D.size()[:2]
        O = Y_C.size(2)
        
        blocks_context = self.context_encoder(X_C, Y_C)
        blocks_target = self.target_encoder(X_D)
        
        output = self.decode(blocks_context, blocks_target)
        output = output.reshape(B, N, O, *X_D.size()[3:])
        
        return output

    def decode(self, blocks_context, blocks_target):
        # pop bottom-level representations for context and target input
        _, h = blocks_context.popitem()
        _, x = blocks_target.popitem()
        
        assert len(x.size()) == 5 and len(h.size()) == 3
        B, M = x.size()[:2]
        T = h.size(1)
        
        # repeat for multi-channel prediction
        x = x.unsqueeze(2).repeat(1, 1, T, 1, 1, 1)
        
        # bottom-level decoding
        x = self.pre_conv1(x, h)
        x = self.up_conv1(x)
        
        for i in range(4):
            _, x_low = blocks_target.popitem()
            _, h_low = blocks_context.popitem()
            x_low = x_low.unsqueeze(2).repeat(1, 1, T, 1, 1, 1)
            x_low = self.pre_convs[i](x_low, h_low)
            x = torch.cat((x, x_low), 3)
            x = self.post_convs[i](x, h_low) # (B, M, T, C, H, W)
            x = self.up_convs[i](x)
                
        x = self.final_conv(x)

        return x

def get_model(model_type='efficientnet-b0'):
    context_encoder = EfficientNet.context_encoder(model_type, pretrained=False, in_channels=4)
    target_encoder = EfficientNet.target_encoder(model_type, pretrained=False)
    model = EfficientWnet(context_encoder, target_encoder)
    return model