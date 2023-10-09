""" Basic modules for generatl U-Net models (1D, 2D, 3D)
modules for 3D-Unets are not done!
"""
import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, reduce


def conv_nd(unet_dim, *args, **kwargs):
    """ Specify Conv for general U-Net """
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[unet_dim](*args, **kwargs)


class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        flattened_weights = rearrange(weight, 'o ... -> o (...)')

        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')

        var = torch.var(flattened_weights, dim = -1, unbiased = False)
        var = rearrange(var, 'o -> o 1 1 1')

        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        flattened_weights = rearrange(weight, "o ... -> o (...)")

        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = torch.var(flattened_weights, dim=-1, unbiased=False)
        var = rearrange(var, "o -> o 1 1 1 1")
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def weight_standardized_conv_nd(unet_dim, *args, **kwargs):
    """ Specify WeightStandardizedConv for general U-Net """
    return {1: WeightStandardizedConv1d, 2: WeightStandardizedConv2d, 3: WeightStandardizedConv3d}[unet_dim](*args, **kwargs)


class Residual(nn.Module):
    """ Residual for 1D U-Net """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module):
    def __init__(self, unet_dim: int, dim):
        super().__init__()

        if unet_dim == 1:
            self.g = nn.Parameter(torch.ones(1, dim, 1))
        elif unet_dim == 2:
            self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        else: # TODO: for 3D U-Net?
            raise NotImplementedError
    
    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class LayerNorm(nn.Module):
    def __init__(self, unet_dim: int, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable

        if unet_dim == 1:
            self.g = nn.Parameter(torch.ones(1, dim, 1))
        elif unet_dim == 2:
            self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        else: # TODO: for 3D U-Net?
            raise NotImplementedError
    
    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable: # ChanLayerNorm from https://github.com/lucidrains/DALLE2-pytorch/blob/main/dalle2_pytorch/dalle2_pytorch.py
            x = x / x.amax(dim=1, keepdim=True).detach()
        
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    """ for 1D U-Net """
    def __init__(self, unet_dim: int, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(unet_dim=unet_dim, dim=dim)
    
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class PreNormCross(nn.Module):
    def __init__(self, unet_dim: int, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(unet_dim=unet_dim, dim=dim)

    def forward(self, x, context):
        x = self.norm(x)
        return self.fn(x, context)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # (D//2,)
        emb = torch.einsum("B, D -> BD", x, emb) # (B, D//2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # (B, D)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Following @crowsonkb's lead with random (learned optional) sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "B -> B 1") # (B, 1)
        freqs = x * rearrange(self.weights, "D -> 1 D") * 2 * math.pi # (1, D//2)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1) # (B, D)
        fouriered = torch.cat((x, fouriered), dim = -1) # (B, D+1)
        return fouriered