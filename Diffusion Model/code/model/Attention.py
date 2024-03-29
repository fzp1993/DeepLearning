import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=1, 
                              num_channels=in_channels, 
                              eps=1e-6, 
                              affine=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    '''
    创建交叉注意力机制，输入的是(batch, n, query_dim)和(batch, n, context_dim)的张量
    需要设置注意力的头数head和每个头的维度dim_head
    '''
    def __init__(self, query_dim, 
                 context_dim=None, 
                 heads=8, 
                 dim_head=64, 
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim == 'None':
            context_dim = None
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # b: batch, n: 特征个数, h: head的个数, d: head的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), 
                      (q, k, v))

        # \frac{QK^\dagger}{\sqrt{d}}
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # softmax(\frac{QK^\dagger}{\sqrt{d}})
        attn = sim.softmax(dim=-1)

        # softmax(\frac{QK^\dagger}{\sqrt{d}})V
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, 
                 n_heads, 
                 d_head, 
                 dropout=0., 
                 context_dim=None, 
                 gated_ff=True, 
                 checkpoint=True):
        super().__init__()
        # attn1是一个自注意力，因为没有设置context_dim
        self.attn1 = CrossAttention(query_dim=dim, 
                                    heads=n_heads, 
                                    dim_head=d_head, 
                                    dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # attn2中的context_dim如果是None，那么attn2就是一个自注意力
        self.attn2 = CrossAttention(query_dim=dim, 
                                    context_dim=context_dim,
                                    heads=n_heads, 
                                    dim_head=d_head, 
                                    dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # def forward(self, x, context=None):
    #     return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    # def _forward(self, x, context=None):
    #     x = self.attn1(self.norm1(x)) + x
    #     x = self.attn2(self.norm2(x), context=context) + x
    #     x = self.ff(self.norm3(x)) + x
    #     return x

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    transformer模块，n_heads是头的个数，d_head是每个头的维度，depth是transformer的堆叠个数
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, 
                                   n_heads, 
                                   d_head, 
                                   dropout=dropout, 
                                   context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
    
if __name__ == '__main__':
    batch = 2
    channel = 3
    height = 5
    width = 6
    n_feature = 11
    dim_context = 10
    input = torch.rand(batch,channel,height,width)
    context = torch.rand(batch,n_feature,dim_context)
    print("origin input's shape: \n batch: %d\n channel: %d\n height: %d\n width: %d\n" 
          %(input.shape))
    print("origin context's shape: \n batch: %d\n n_feature: %d\n dim_context: %d\n" 
          %(context.shape))
    n_heads = 8
    d_head = 7
    depth = 9
    dropout = 0.
    model = SpatialTransformer(in_channels=channel, 
                               n_heads=n_heads, 
                               d_head=d_head,
                               depth=depth, 
                               dropout=dropout, 
                               context_dim=dim_context)
    print("SpatialTransformer's parameters: \n num_heads: %d\n dim_head: %d\n depth: %d\n dropout: %d\n" 
          %(n_heads, d_head, depth, dropout))
    y = model(input,context)