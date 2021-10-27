from torch.hub import load_state_dict_from_url
from .utils import *
from collections import OrderedDict
from .attention import SAB


class EfficientNet(nn.Module):

    def __init__(self, block_args_list, global_params, in_channels=3):
        super().__init__()

        self.block_args_list = block_args_list
        self.global_params = global_params

        # Batch norm parameters
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon

        # Stem
        out_channels = round_filters(32, self.global_params)
        self._conv_stem = Conv2dSamePadding(in_channels,
                                            out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            bias=False,
                                            name='stem_conv')
        self._bn0 = BatchNorm2d(num_features=out_channels,
                                momentum=batch_norm_momentum,
                                eps=batch_norm_epsilon,
                                name='stem_batch_norm')

        self._swish = Swish(name='swish')

        # Build _blocks
        idx = 0
        self._blocks = nn.ModuleList([])
        for block_args in self.block_args_list:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
            idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)

            # The rest of the _blocks
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
                idx += 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self.global_params)
        self._conv_head = Conv2dSamePadding(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            bias=False,
                                            name='head_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                                momentum=batch_norm_momentum,
                                eps=batch_norm_epsilon,
                                name='head_batch_norm')

        # Final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)

    def forward(self, x):
        # Stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)
            x = block(x, drop_connect_rate)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Pooling and Dropout
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Fully-connected layer
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, *, n_classes=1000, pretrained=False, in_channels=3):
        return _get_model_by_name(model_name, classes=n_classes, pretrained=pretrained, in_channels=in_channels)

    @classmethod
    def encoder(cls, model_name, *, pretrained=False, in_channels=3):
        model = cls.from_name(model_name, pretrained=pretrained, in_channels=in_channels)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.name = model_name

                self.global_params = model.global_params

                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name='stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name='head_swish')

            def forward(self, x):
                shapes = set()
                blocks = OrderedDict()
                
                # Stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # Blocks
                count = 0
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)
                    
                    shape = x.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[block._bn2.name] = x

                # Head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                
                blocks.popitem()
                blocks[self.head_swish.name] = x
                
                return blocks

        return Encoder()

    @classmethod
    def target_encoder(cls, model_name, *, pretrained=False, in_channels=3):
        model = cls.from_name(model_name, pretrained=pretrained, in_channels=in_channels)

        class TargetEncoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.name = model_name

                self.global_params = model.global_params

                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name='stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name='head_swish')

            def forward(self, X):
                shapes = set()
                blocks = OrderedDict()
                
                assert len(X.size()) == 5
                B, N, K, H, W = X.size()
                x = X.reshape(B*N, K, H, W)
                
                # Stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # Blocks
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)
                    
                    K, H, W = x.size()[-3:]
                    if (H, W) not in shapes:
                        blocks[block._bn2.name] = x.reshape(B, N, K, H, W)
                        shapes.add((H, W))

                # Head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                
                blocks.popitem()
                K, H, W = x.size()[-3:]
                blocks[self.head_swish.name] = x.reshape(B, N, K, H, W)
                
                return blocks

        return TargetEncoder()

    @classmethod
    def context_encoder(cls, model_name, *, pretrained=False, in_channels=4):
        model = cls.from_name(model_name, pretrained=pretrained, in_channels=in_channels)

        class ContextEncoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.name = model_name

                self.global_params = model.global_params

                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name='stem_swish')
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name='head_swish')
                
                self.channel_attentions = nn.ModuleList([SAB(self.size[3-i]) for i in range(4)] + [SAB(self.n_channels)])
                self.example_attentions = nn.ModuleList([SAB(self.size[3-i]) for i in range(4)] + [SAB(self.n_channels)])
                self.example_mlps = nn.ModuleList([nn.Sequential(nn.Linear(self.size[3-i], self.size[3-i]),
                                                                 nn.GELU(),
                                                                 nn.Linear(self.size[3-i], self.size[3-i]))
                                                   for i in range(4)] + \
                                                  [nn.Sequential(nn.Linear(self.n_channels, self.n_channels),
                                                                 nn.GELU(),
                                                                 nn.Linear(self.n_channels, self.n_channels))])
                self.merge_convs = nn.ModuleList([nn.Conv2d(2*self.size[3-i], self.size[3-i], kernel_size=3, stride=1, padding=1)
                                                  for i in range(4)] + \
                                                 [nn.Conv2d(2*self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1)])

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
                return size_dict[self.name]
            
            @property
            def n_channels(self):
                n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                                   'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                                   'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
                return n_channels_dict[self.name]

            def forward(self, X_C, Y_C):
                shapes = set()
                blocks = OrderedDict()
                
                assert len(X_C.size()) == 5 and len(Y_C.size()) == 5
                B, N, K, H, W = X_C.size()
                B, N, T, H, W = Y_C.size()
                C = torch.stack([torch.cat((X_C, Y_C[:, :, t].unsqueeze(2)), 2)
                                 for t in range(T)], 2) # (B, N, T, K+1, H, W)
                
                x = C.reshape(B*N*T, K+1, H, W)
                
                # Stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # Blocks
                count = 0
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)
                    
                    K, H, W = x.size()[-3:]
                    if (H, W) not in shapes and count < 4:
                        x = x.reshape(B, N, T, K, H, W).permute(0, 1, 4, 5, 2, 3).reshape(B*N*H*W, T, K)
                        x = self.channel_attentions[count](x)
                        
                        x = x.reshape(B, N, H, W, T, K).transpose(1, 4).reshape(B*T*H*W, N, K)
                        x = self.example_attentions[count](x)

#                         # reversed
#                         x = x.reshape(B, N, T, K, H, W).permute(0, 2, 4, 5, 1, 3).reshape(B*T*H*W, N, K)
#                         x = self.example_attentions[count](x)
                        
#                         x = x.reshape(B, T, H, W, N, K).transpose(1, 4).reshape(B*N*H*W, T, K)
#                         x = self.channel_attentions[count](x)

#                         # parallel
#                         x1 = x.reshape(B, N, T, K, H, W).permute(0, 1, 4, 5, 2, 3).reshape(B*N*H*W, T, K)
#                         x1 = self.channel_attentions[count](x1)
                        
#                         x2 = x.reshape(B, N, H, W, T, K).transpose(1, 4).reshape(B*T*H*W, N, K)
#                         x2 = self.example_attentions[count](x2)
                        
#                         x1 = x1.reshape(B, N, H, W, T, K).permute(0, 1, 4, 5, 2, 3).reshape(B*N*T, K, H, W)
#                         x2 = x2.reshape(B, T, H, W, N, K).permute(0, 4, 1, 5, 2, 3).reshape(B*N*T, K, H, W)
#                         x = torch.cat((x1, x2), 1)
#                         x = self.merge_convs[count](x)
                        
                        h = x.mean(2).mean(2).reshape(B, N, T, K)
                        h = self.example_mlps[count](h)
                        h = h.mean(1)
                        blocks[block._bn2.name] = h
                        
                        shapes.add((H, W))
                        count += 1
                        
                K, H, W = x.size()[-3:]
                x = x.reshape(B, N, T, K, H, W).reshape(B*N*T, K, H, W)

                # Head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                
                K, H, W = x.size()[-3:]
                x = x.reshape(B, N, T, K, H, W).permute(0, 1, 4, 5, 2, 3).reshape(B*N*H*W, T, K)
                x = self.channel_attentions[4](x)

                x = x.reshape(B, N, H, W, T, K).transpose(1, 4).reshape(B*T*H*W, N, K)
                x = self.example_attentions[4](x)

                x = x.reshape(B, T, H, W, N, K).permute(0, 4, 1, 5, 2, 3).reshape(B*N*T, K, H, W)

                h = x.mean(2).mean(2).reshape(B, N, T, K)
                h = self.example_mlps[4](h)
                h = h.mean(1)
                blocks[self.head_swish.name] = h
                
                return blocks

        return ContextEncoder()

    @classmethod
    def custom_head(cls, model_name, *, n_classes=1000, pretrained=False):
        if n_classes == 1000:
            return cls.from_name(model_name, n_classes=n_classes, pretrained=pretrained)
        else:
            class CustomHead(nn.Module):
                def __init__(self, out_channels):
                    super().__init__()
                    self.encoder = cls.encoder(model_name, pretrained=pretrained)
                    self.custom_head = custom_head(self.n_channels * 2, out_channels)

                @property
                def n_channels(self):
                    n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                                       'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                                       'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
                    return n_channels_dict[self.encoder.name]

                def forward(self, x):
                    x = self.encoder(x)
                    mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
                    ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
                    x = torch.cat([mp, ap], dim=1)
                    x = x.view(x.size(0), -1)
                    x = self.custom_head(x)

                    return x

            return CustomHead(n_classes)


def _get_model_by_name(model_name, classes=1000, pretrained=False, in_channels=3):
    block_args_list, global_params = get_efficientnet_params(model_name, override_params={'num_classes': classes})
    model = EfficientNet(block_args_list, global_params, in_channels=in_channels)
    try:
        if pretrained:
            pretrained_state_dict = load_state_dict_from_url(IMAGENET_WEIGHTS[model_name])
            
            if in_channels != 3:
                random_state_dict = model.state_dict()
                pretrained_state_dict['_conv_stem.weight'] = random_state_dict['_conv_stem.weight']

            if classes != 1000:
                random_state_dict = model.state_dict()
                pretrained_state_dict['_fc.weight'] = random_state_dict['_fc.weight']
                pretrained_state_dict['_fc.bias'] = random_state_dict['_fc.bias']

            model.load_state_dict(pretrained_state_dict)

    except KeyError as e:
        print(f"NOTE: Currently model {e} doesn't have pretrained weights, therefore a model with randomly initialized"
              " weights is returned.")

    return model
