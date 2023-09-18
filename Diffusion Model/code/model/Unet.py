import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from model.Attention import SpatialTransformer
from utils.util import (conv_nd, 
                        avg_pool_nd, 
                        normalization, 
                        linear,
                        zero_module,
                        timestep_embedding)

class TimestepBlock(nn.Module):

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    '''
    上采样结构，对原始数据进行插值
    '''
    def __init__(self, channels, 
                 use_conv, 
                 dims = 2, 
                 out_channels = None, 
                 padding = 1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, 
                                self.channels, 
                                self.out_channels, 
                                kernel_size=3,
                                padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    '''
    下采样结构
    '''
    def __init__(self, channels, 
                 use_conv, 
                 dims = 2, 
                 out_channels = None,
                 padding = 1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, 
                              self.channels, 
                              self.out_channels, 
                              kernel_size=3, 
                              stride=stride, 
                              padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    '''
    残差模块
    '''
    def __init__(self,
                 channels,
                 timestep_emb_channels,
                 dropout,
                 out_channels=0,
                 use_conv=True,
                 use_scale_shift_norm=False,
                 dims=2,
                 up=False,
                 down=False):
        super().__init__()
        self.channels = channels
        self.timestep_emb_channels = timestep_emb_channels
        self.dropout = dropout
        if out_channels != 0:
            self.out_channels = out_channels
        else:
            self.out_channels = channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, 
                    channels, 
                    self.out_channels, 
                    kernel_size=3,
                    padding=1))

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.timestep_emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                timestep_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, 
                        self.out_channels, 
                        self.out_channels, 
                        kernel_size=3,
                        padding=1)))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, 
                                           channels, 
                                           self.out_channels, 
                                           kernel_size=3,
                                           padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, 
                                           channels, 
                                           self.out_channels, 
                                           kernel_size=3)
    
    def forward(self, x, timestep_emb):
        if self.updown:
            in_rest = self.in_layers[:-1] # normalization正则化层和SiLU激活层
            in_conv = self.in_layers[-1] # conv_nd卷积层
            h = in_rest(x)
            h = self.h_upd(h) # 对h进行一次上或下采样
            x = self.x_upd(x) # 对x进行一次上或下采样
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        timestep_emb_out = self.timestep_emb_layers(timestep_emb).type(h.dtype)
        while len(timestep_emb_out.shape) < len(h.shape):
            timestep_emb_out = timestep_emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm = self.out_layers[0] # normalization正则化层
            out_rest = self.out_layers[1:] # SiLU激活层、linear线性层、Dropout层和conv_nd卷积层
            scale, shift = torch.chunk(timestep_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + timestep_emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModel(nn.Module):
    '''
    in_channels: 输入的channel个数，是一个int类型的数值，例如3
    model_channels: 上下采样中CNN的输出channel的基础值，每个上
                    下采样层的channel都在model_channels的基础
                    上加倍，是一个int类型的数值，例如64
    num_res_blocks: 上下采样中每层的残差结构数量，是一个int类型
                    的数值，例如1
    attention_resolutions: 上下采样中第ds*2层是否添加spatial transformer，
                           是一个set、list或tuple结构，例如[1, 2, 4]
    dropout: dropout率，是一个float类型的数值，例如0.
    channel_mult: 是一个tuple结构，例如(1, 2, 3, 8)，元素个数
                  决定上下采样的层数，元素*model_channels表示
                  每层CNN输出的channel大小
    conv_resample: 是一个bool类型的数值，例如True。如果是True，
                   那么在每层上下采样时如果最后一个模块不使用残
                   差结构，即resblock_updown=False时，就会使用
                   卷积结构
    dims: 判断使用的卷积结构的维度，是一个int类型的数据，取值是1、
          2、3。例如dims=2时，表示使用的卷积结构是2D
    num_classes: 如果是分类任务，表示类别个数，是一个int类型的数
                 值，例如2或None
    num_heads: transformer的头个数，是一个int值，如果不设置这个参
               数，那么赋值-1。它的优先级低于num_head_channels，
               即划分每个channel时的头维度
    num_head_channels: transformer的头维度，是一个int值，如果不
                       设置这个参数，那么赋值-1。需要注意的是，
                       num_heads和num_head_channels不能同时为
                       -1
    use_scale_shift_norm: 主要决定残差层的结构，是一个bool类型的
                          数值，例如True
    resblock_updown: 上下采样中最后一个模块是否需要残差结构，是
                     一个bool类型数据，例如True
    transformer_depth: transformer的堆叠深度，是一个int值，例如
                       1
    context_dim: spatial transformer中cross attention的context
                 特征维度，是一个int值，例如10
    n_embed: 最后输出时CNN的输出channel个数，是一个int值，例如3
    '''
    def __init__(self,
                 in_channels,
                 model_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 num_classes=0,
                 num_heads=-1,
                 num_head_channels=-1,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 transformer_depth=1,
                 context_dim=None,
                 n_embed=0):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = list(map(int,eval(attention_resolutions)))
        self.dropout = dropout
        self.channel_mult = list(map(int,eval(channel_mult)))
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes != 0:
            self.label_emb = nn.Embedding(int(num_classes), time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult): # 获取每次下采样时的层数和channel倍数
            for _ in range(num_res_blocks):
                # 下采样过程中的每一层包括num_res_blocks个layers
                # 每个layers包括一个残差层、一个SpatialTransformer层和一个残差层
                layers = [
                    ResBlock(ch,
                             time_embed_dim,
                             dropout,
                             out_channels=int(mult) * model_channels,
                             dims=dims,
                             use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in self.attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, 
                                           num_heads, 
                                           dim_head, 
                                           depth=transformer_depth, 
                                           context_dim=context_dim))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,
                                 time_embed_dim,
                                 dropout,
                                 out_channels=out_ch,
                                 dims=dims,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 down=True)
                        if resblock_updown
                        else Downsample(ch, 
                                        conv_resample, 
                                        dims=dims, 
                                        out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch, 
                               num_heads, 
                               dim_head, 
                               depth=transformer_depth, 
                               context_dim=context_dim),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in self.attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, 
                                           num_heads, 
                                           dim_head, 
                                           depth=transformer_depth, 
                                           context_dim=context_dim)
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
        )

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes != 0:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(torch.float32)

        return self.id_predictor(h)

if __name__ == '__main__':
    batch = 2
    channel = 1
    height = 32
    width = 32
    n_feature = 11
    dim_context = 10
    input = torch.rand(batch,channel,height,width)
    context = torch.rand(batch,n_feature,dim_context)
    print("origin input's shape: \n batch: %d\n channel: %d\n height: %d\n width: %d\n" 
          %(input.shape))
    print("origin context's shape: \n batch: %d\n n_feature: %d\n dim_context: %d\n" 
          %(context.shape))
    model_channels = 64
    num_res_blocks = 1
    attention_resolutions = [1]
    num_heads = 8
    n_embed = 4
    model = UNetModel(in_channels = channel,
                      model_channels = model_channels,
                      num_res_blocks = num_res_blocks,
                      attention_resolutions = attention_resolutions,
                      dropout=0,
                      channel_mult=(1, 2, 4, 8),
                      conv_resample=True,
                      dims=2,
                      num_classes=0,
                      num_heads=num_heads,
                      num_head_channels=-1,
                      use_scale_shift_norm=False,
                      resblock_updown=True,
                      transformer_depth=4,
                      context_dim=None,
                      n_embed=n_embed
                      )
    y = model(x = input,timesteps = torch.rand(2),context = None)
    print(y.shape)