""" Modules for general U-Net models (1D, 2D, 3D)
from https://github.com/lucidrains/denoising-diffusion-pytorch
modules for 3D-Unets are not done!
"""
from functools import partial

import torch
from torch import nn
from einops import rearrange

from modules  import default, cast_tuple
from modules.unet.base_modules import conv_nd, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, \
    Residual, PreNorm
from modules.unet.unet_modules import attention_nd, linear_attention_nd, ResnetBlock, \
    Downsample, Upsample


class UnetBase(nn.Module):
    """ Base of the general U-Net (1D, 2D, 3D(todo)) """
    def __init__(
            self,
            unet_dim: int,
            dim,
            init_dim = None,
            out_dim = None,
            dim_mults = (1, 2, 4, 8),
            channels = 3,
            self_condition = False,
            resnet_groups = 8,
            learned_variance = False,
            learned_sinusoidal_cond = False,
            random_fourier_features = False,
            learned_sinusoidal_dim = 16,
            sinusoidal_pos_emb_theta = 10000,
            attn_dim_head = 32,
            attn_heads = 4,
            full_attn = (False, False, False, True) # length should be equals to length of dim_mults
    ):
        """ General U-Net model

        Args:
            unet_dim (int): Dimension of U-Net model. (1D, 2D, 3D(todo))
            dim (int): Base dimension for U-Net.
            channels (int): Channel dimension for input and output.
            init_dim (int, optional): If given, use it as an initial dimension. Defaults to None.
            out_dim (int, optional): If given, use it as an output dimension. Defaults to None.
            dim_mults (tuple): Elements to be multiplied by the dim. Defaults to (1, 2, 4, 8).
            full_attn (tuple or list, optional): All False for 1D UNet. Defaults to (False, False, False, True) (2D)
        """
        super().__init__()

        # determine dimensions
        self.unet_dim = unet_dim
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = conv_nd(unet_dim, input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, unet_dim=self.unet_dim, groups=resnet_groups)

        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinusoidal_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinusoidal_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim
        
        self.time_mlp = nn.Sequential(
            sinusoidal_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention settings
        num_stages = len(dim_mults)
        full_attn_tuple = cast_tuple(full_attn, num_stages)
        attn_heads_tuple = cast_tuple(attn_heads, num_stages)
        attn_dim_head_tuple = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn_tuple) == len(dim_mults)
        
        FullAttention = partial(attention_nd(unet_dim=self.unet_dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_res = len(in_out)
        
        # down sampling part
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(in_out, full_attn_tuple, attn_heads_tuple, attn_dim_head_tuple)):
            is_last = ind >= (num_res - 1)

            attn_class = FullAttention if layer_full_attn else linear_attention_nd(unet_dim=self.unet_dim)

            self.downs.append(nn.ModuleList([
                block_class(dim_in, dim_in, time_cond_dim=time_dim),
                block_class(dim_in, dim_in, time_cond_dim=time_dim),
                attn_class(dim_in, heads=layer_attn_heads, dim_head=layer_attn_dim_head),
                Downsample(dim_in, dim_out=dim_out, unet_dim=self.unet_dim) if not is_last else conv_nd(self.unet_dim, dim_in, dim_out, 3, padding=1)
            ]))

        # middle part
        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_cond_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
        self.mid_block2 = block_class(mid_dim, mid_dim, time_cond_dim=time_dim)

        # up sampling part
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(*map(reversed, (in_out, full_attn_tuple, attn_heads_tuple, attn_dim_head_tuple)))):

            is_last = ind == (len(in_out) - 1)

            attn_class = FullAttention if layer_full_attn else linear_attention_nd(unet_dim=self.unet_dim)

            self.ups.append(nn.ModuleList([
                block_class(dim_out+dim_in, dim_out, time_cond_dim=time_dim),
                block_class(dim_out+dim_in, dim_out, time_cond_dim=time_dim),
                attn_class(dim_out, heads=layer_attn_heads, dim_head=layer_attn_dim_head),
                Upsample(dim_out, dim_out=dim_in, unet_dim=self.unet_dim) if not is_last else conv_nd(self.unet_dim, dim_out, dim_in, 3, padding=1)
            ]))

        # final normalization part
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_class(dim*2, dim, time_cond_dim=time_dim)
        self.final_conv = conv_nd(unet_dim, dim, self.out_dim, 1)
    
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)
    
    def forward(self, x, time, context=None, context_cross=None, x_self_cond = None):
        assert all([(d % self.downsample_factor) == 0 for d in x.shape[-2:]]), "Input dimensions {} need to be devidible by {}.".format(x.shape[-2:], self.downsample_factor)

        if context is not None or context_cross is not None:
            raise NotImplementedError("Conditioning should be implemented additionally.")

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

