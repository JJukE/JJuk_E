""" Modules for general U-Net models (1D, 2D, 3D)
from https://github.com/lucidrains/denoising-diffusion-pytorch
modules for 3D-Unets are not done!
"""
import torch
from torch import nn
from einops import rearrange, reduce

from jjuke.modules.unet.base_modules import conv_nd, weight_standardized_conv_nd, \
    RMSNorm, LayerNorm


class Block(nn.Module):
    def __init__(self, unet_dim: int, dim, dim_out, groups = 8, weight_standardization = False):
        super().__init__()
        
        if not weight_standardization:
            self.proj = conv_nd(unet_dim, dim, dim_out, 3, padding=1)
        else:
            self.proj = weight_standardized_conv_nd(unet_dim, dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *args, unet_dim=None, time_emb_dim=None, groups=8, weight_standardization=False):
        super().__init__()

        self.unet_dim = unet_dim

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None
        self.block1 = Block(unet_dim, dim, dim_out, groups=groups, weight_standardization=weight_standardization)
        self.block2 = Block(unet_dim, dim_out, dim_out, groups=groups, weight_standardization=weight_standardization)
        self.res_conv = conv_nd(unet_dim, dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            if self.unet_dim == 1:
                time_emb = rearrange(time_emb, "B C -> B C 1")
            elif self.unet_dim == 2:
                time_emb = rearrange(time_emb, "B C -> B C 1 1")
            else: # TODO: for 3D U-Net?
                raise NotImplementedError
            scale_shift = time_emb.chunk(2, dim=1) # (B, C/2, ...), (B, C/2, ...)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention1D(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(1, dim)
        self.to_qkv = conv_nd(1, dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Sequential(
            conv_nd(1, hidden_dim, dim, 1),
            LayerNorm(1, dim)
        )

    def forward(self, x):
        b, c, n = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class LinearAttention2D(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(2, dim)
        self.to_qkv = conv_nd(2, dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Sequential(
            conv_nd(2, hidden_dim, dim, 1),
            LayerNorm(2, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x) # for 2D U-Net
        
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


def linear_attention_nd(unet_dim):
    """ Specify LinearAttention for general U-Net """
    return {1: LinearAttention1D, 2: LinearAttention2D}[unet_dim]


class Attention1D(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(1, dim)

        self.to_qkv = conv_nd(1, dim, hidden_dim*3, 1, bias=False)
        self.to_out = conv_nd(1, hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)
        

class Attention2D(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(2, dim)

        self.to_qkv = conv_nd(2, dim, hidden_dim*3, 1, bias=False)
        self.to_out = conv_nd(2, hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


def attention_nd(unet_dim):
    """ Specify Attention for general U-Net """
    return {1: Attention1D, 2: Attention2D}[unet_dim]