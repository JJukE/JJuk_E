""" Modules for general U-Net models (1D, 2D, 3D)
from https://github.com/lucidrains/denoising-diffusion-pytorch
modules for 3D-Unets are not done!
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from modules.unet.base_modules import conv_nd, weight_standardized_conv_nd, \
    RMSNorm, LayerNorm
from modules import default, resize_feature_to


def Downsample(dim, dim_out = None, unet_dim = None):
    # TODO: check when 1D -> return conv_nd(unet_dim, default(dim_out, dim), 4, 2, 1) in lucidrain ddpm
    return nn.Sequential(
        Rearrange("B C (H P1) (W P2) -> B (C P1 P2) H W", P1=2, P2=2),
        conv_nd(unet_dim, dim*4, default(dim_out, dim), 1)
    )


def Upsample(dim, dim_out = None, unet_dim = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv_nd(unet_dim, dim, default(dim_out, dim), 3, padding=1)
    )


class UpsampleCombiner(nn.Module):
    def __init__(self, dim, *, enabled = False, dim_ins = tuple(), dim_outs = tuple()):
        super().__init__()
        assert len(dim_ins) == len(dim_outs)
        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.feature_convs = nn.ModuleList([
            Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)
        ])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, features = None):
        target_size = x.shape[-1] # 2D -> h or w // 1D -> n

        features = default(features, tuple())

        if not self.enabled or len(features) == 0 or len(self.feature_convs) == 0:
            return x
        
        features = [resize_feature_to(feature, target_size) for feature in features]
        outs = [conv(feature) for feature, conv in zip(features, self.feature_convs)]
        return torch.cat((x, *outs), dim=1) # channel dimension
        


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
    def __init__(
            self,
            dim,
            dim_out,
            *args,
            unet_dim = None,
            cond_dim = None,
            time_cond_dim = None,
            groups = 8,
            weight_standardization = False,
            cosine_sim_cross_attn = False
    ):
        super().__init__()

        self.unet_dim = unet_dim

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_cond_dim, dim_out * 2)
        ) if time_cond_dim is not None else None

        self. cross_attn = CrossAttention(
            dim = dim_out,
            context_dim = cond_dim,
            cosine_dim = cosine_sim_cross_attn
        ) if cond_dim is not None else None
        
        self.block1 = Block(unet_dim, dim, dim_out, groups=groups, weight_standardization=weight_standardization)
        self.block2 = Block(unet_dim, dim_out, dim_out, groups=groups, weight_standardization=weight_standardization)
        self.res_conv = conv_nd(unet_dim, dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            if self.unet_dim == 1:
                time_emb = rearrange(time_emb, "B C -> B C 1")
            elif self.unet_dim == 2:
                time_emb = rearrange(time_emb, "B C -> B C 1 1")
            else: # TODO: for 3D U-Net?
                raise NotImplementedError
            scale_shift = time_emb.chunk(2, dim=1) # (B, C/2, ...), (B, C/2, ...)
        
        h = self.block1(x, scale_shift=scale_shift)

        if self.cross_attn is not None:
            assert cond is not None
            
            h = rearrange(h, "B C ... -> B ... C")
            h, ps = pack([h], "B * C")

            h = self.cross_attn(h, context=cond) + h

            h, = unpack(h, ps, "B * C")
            h = rearrange(h, "B ... C -> B C ...")

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


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim, dim_out, kernel_sizes, unet_dim=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t%2) == (stride%2), kernel_sizes)])

        self.unet_dim = unet_dim

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel_size, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(conv_nd(unet_dim, dim, dim_scale, kernel_size,
                                      stride=stride, padding=(kernel_size-stride)//2))
    
    def forward(self, x):
        feats = tuple(map(lambda conv: conv(x), self.convs)) # NOTE: (B C N) or (B C h w)
        return torch.cat(feats, dim=1)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            *args,
            heads = 8,
            dim_head = 64,
            dropout = 0.,
            rotary_emb = None,
            cosine_sim = True,
            cosine_sim_scale = 16
    ):
        super().__init__()
        
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head*2, bias=False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, mask=None):
        B, N = x.shape[:2]

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)) # TODO: check shape

        q = rearrange(q, "B N (H d) -> B H N d", H=self.heads)
        q = q * self.scale

        # rotary embeddings
        if self.rotary_emb is not None:
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))
        
        # cosine similarity
        if self.cosine_sim:
            q, k = map(lambda t: F.normalize(t, dim=-1), (q, k))
        
        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))
        
        # calculate query, key similarities
        sim = torch.einsum("B H i d, B j d -> B H i j", q, k)
        
        # masking
        max_neg_value = -torch.finfo(sim.dtype).max # finfo: range of the datatype

        if mask is not None:
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "B j -> B 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)
        
        # attention
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)
        attn = self.dropout(attn)

        # aggregate values
        out = torch.einsum("B H i j, B j d -> B H i d", attn, v)
        out = rearrange(out, "B H N d -> B N (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, unet_dim: int, dim, heads = 8, dim_head = 32, **kwargs):
        super().__init__()

        self.unet_dim = unet_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm(unet_dim, dim)
        self.nonlinear = nn.GELU()
        self.to_qkv = conv_nd(dim, hidden_dim*3, 1, bias=False)

        self.to_out = nn.Sequential(
            conv_nd(unet_dim, hidden_dim, dim, 1, bias=False),
            LayerNorm(unet_dim, dim)
        )
    
    def forward(self, feature):
        H = self.heads

        if self.unet_dim == 1:
            D = feature.shape[-1]
            seq_len = D
        elif self.unet_dim == 2:
            h, w = feature.shape[-2:] # featue map
            seq_len = h * w
        else:
            raise NotImplementedError

        feature = self.norm(feature) # (B, C, D) or (B, C, h, w)
        q, k, v = self.to_qkv(feature).chunk(3, dim=1) # (B, H*h_dim, d) or (B, H*h_dim, h, w)
        # TODO: Try to deal with 1D and 2D at once!
        # q, k, v = map(lambda t: rearrange(t, "B (H c) ... -> (B H) ... c", H=H), (q, k, v))
        # (q, ps_q), (k, ps_k), (v, ps_v) = map(lambda t: pack([t], "b * c"), (q, k, v))

        if self.unet_dim == 1:
            q, k, v = map(lambda t: rearrange(t, "B (H c) D -> (B H) D c", H=H), (q, k, v)) # (B*H=b, seq_len(=D=N), h_dim)
        elif self.unet_dim == 2:
            q, k, v = map(lambda t: rearrange(t, "B (H c) h w -> (B H) (h w) c", H=H), (q, k, v)) # (B*H=b, seq_len(=N), h_dim)

        q = q.softmax(dim=-1) # along the channel
        k = k.softamx(dim=-2) # along the sequence length

        q = q * self.scale
        v = F.normalize(v, dim=-1) # along the channel

        k, v = map(lambda t: t / math.sqrt(seq_len), (k, v))

        context = torch.einsum("b N d, b N e -> b d e", k, v) # (B*H, key_dim, value_dim)
        out = torch.einsum("b N d, b d e -> b N e", q, context) # (B*H, seq_len, value_dim)
        if self.unet_dim == 1:
            out = rearrange(out, "(B H) N e -> B (H e) N", H=H, N=D) # (B, H*h_dim, D)
        elif self.unet_dim == 2:
            out = rearrange(out, "(B H) (h w) e -> B (H e) w h", H=H, h=h, w=w) # (B, H*h_dim, h, w)
        return self.to_out(out) # (B, C, D) or (B, C, h, w)


class CrossAttention(nn.Module):
    def __init__(
            self,
            unet_dim: int,
            dim, 
            *args,
            context_dim = None,
            dim_head = 64,
            heads = 8,
            dropout = 0.,
            norm_context = False,
            cosine_sim = False,
            cosine_sim_scale = 16
    ):
        super().__init__()

        self.cosine_sim = cosine_sim
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(unet_dim, dim)
        self.norm_context = LayerNorm(unet_dim, context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, hidden_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias=False),
            LayerNorm(unet_dim, dim)
        )

    def forward(self, x, context, mask=None):
        H = self.heads
        B, N = x.shape[:2]

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "B N (H d) -> B H N d", H=H), (q, k, v))

        # cosine similarity
        if self.cosine_dim:
            q, k = map(lambda t: F.normalize(t, dim=-1), (q, k))
        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query, key similarities
        sim = torch.einsum("B H i d, B H j d -> B H i j", q, k)

        # masking
        max_neg_value = -torch.finfo(sim.dtype).max
        if mask is not None:
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "B j -> B 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)
        
        # attention
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        out = torch.einsum("B H i j, B H j d -> B H i d", attn, v)
        out = rearrange(out, "B H N d -> B N (H d)")
        return self.to_out(out)
            