""" Modules for general U-Net models (1D, 2D, 3D)
from https://github.com/lucidrains/DALLE2-pytorch (added some features from UnetBase)
Resume after studying classifier free guidance..!!
"""
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from jjuke.modules  import default, cast_tuple
from jjuke.modules.unet.base_modules import conv_nd, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, \
    Residual, PreNorm
from jjuke.modules.unet.unet_modules import attention_nd, linear_attention_nd, ResnetBlock, CrossEmbedLayer, \
    Downsample, Upsample, UpsampleCombiner


class RearrangeToSequence(nn.Module):
    """ Pack along size(2D: h*w, 1D: n) -> Apply function -> Unpack """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        x = rearrange(x, "B C ... -> B ... C")
        x, ps = pack([x], "B * C")

        x = self.fn(x)

        x, = unpack(x, ps, "B * C")
        x = rearrange(x, "B ... C -> B C ...")
        return x


class CondUnet(nn.Module):
    """ Base of the Conditional U-Net (1D, 2D, 3D(todo)) """
    def __init__(
            self,
            unet_dim: int,
            dim,
            conditioning = None, # ["text", "img", "..."] # TODO: image and etc.?
            text_embed_dim = None,
            cond_dim = None,
            num_time_tokens = 2,
            init_dim = None,
            out_dim = None,
            dim_mults = (1, 2, 4, 8),
            channels = 3,
            in_channels = None,
            out_channels = None,
            self_cond = False,
            init_conv_kernel_size = 7,
            final_conv_kernel_size = 1,
            num_resnet_blocks = 2,
            resnet_groups = 8,
            resnet_weight_standardization = False,
            sparse_attn = False,
            attn_heads = 4,
            attn_dim_head = 32,
            cosine_sim_cross_attn = False,
            cosine_sim_self_attn = False,
            init_cross_embed = False,
            init_cross_embed_kernel_sizes = (3, 7, 15),
            downsample_cross_embed = False,
            downsample_cross_embed_kernel_sizes = (2, 4),
            combine_upsample_features = False,
            memory_efficient = False,
            checkpoint_during_training = False,
    ):
        """ Conditional U-Net model

        Args:
            unet_dim (int): Dimension of U-Net model. (1D, 2D, 3D(todo))
            dim (int): Base dimension for U-Net.
            cond_dim (int): Dimension of condition embedding.
            channels (int): Channel dimension for input and output.
            init_dim (int, optional): If given, use it as an initial dimension. Defaults to None.
            out_dim (int, optional): If given, use it as an output dimension. Defaults to None.
            dim_mults (tuple): Elements to be multiplied by the dim. Defaults to (1, 2, 4, 8).
            self_cond (bool): Set this True to use the self-conditioning technique from - https://arxiv.org/abs/2208.04202
        """
        super().__init__()

        # determine dimensions
        self.unet_dim = unet_dim
        self.conditioning = conditioning
        self.channels = channels
        self.self_cond = self_cond
        self.in_channels = default(in_channels, channels) * (2 if self_cond else 1) # variable
        self.out_channels = default(out_channels, channels) # variable

        self.init_dim = default(init_dim, dim)

        if init_cross_embed: # TODO: check if it works in 1D
            self.init_conv = CrossEmbedLayer(unet_dim, in_channels, init_dim, kernel_sizes=init_cross_embed_kernel_sizes, stride=1)
        else:
            self.init_conv = conv_nd(unet_dim, in_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size//2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # [(init_dim, dim_1), (dim_1, dim_2), ...]
        num_stages = len(in_out)

        if conditioning is not None:
            assert conditioning in "text|img".split("|")
            # time, image embeddings, and optional text encoding
            cond_dim = default(cond_dim, dim)
            time_cond_dim = dim * 4

            self.to_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_cond_dim),
                nn.GeLU()
            )

            self.to_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim*num_time_tokens),
                Rearrange("B (N D) -> B N D", N=num_time_tokens) # TODO: Check if it works in 1D (maybe works!)
            )

            self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

            self.norm_cond = nn.LayerNorm(unet_dim, cond_dim)
            self.norm_mid_cond = nn.LayerNorm(unet_dim, cond_dim)

            if conditioning == "text":
                assert text_embed_dim is not None, "text_embed_dim must be given if conditioning with text!"
                self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
                self.text_embed_dim = text_embed_dim
            
        # attention parameters
        attn_kwargs = dict(heads = attn_heads, dim_head=attn_dim_head, cosine_sim=cosine_sim_self_attn)
        self_attn = cast_tuple(self_attn, num_stages)
        create_self_attn = lambda dim: RearrangeToSequence(Residual(Attention(dim, **attn_kwargs)))

        # resnet block class
        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = None if len(resnet_groups) == 0 else resnet_groups[0]
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)
        resnet_block = partial(ResnetBlock, unet_dim=unet_dim,
                               cosine_sim_cross_attn=cosine_sim_cross_attn, weight_standardization=resnet_weight_standardization)

        # downsample class
        downsample_class = partial(Downsample, unet_dim=unet_dim)
        if downsample_cross_embed:
            cownsample_class = partial(CrossEmbedLayer, unet_dim=unet_dim,
                                       kernel_sizes=downsample_cross_embed_kernel_sizes)
        
        # upsample class
        upsample_class = partial(Upsample, unet_dim=unet_dim)
        
        # give memory efficient unet an initial resnet block
        if memory_efficient:
            self.init_resnet_block = resnet_block(init_dim, init_dim, time_cond_dim=time_cond_dim, groups=top_level_resnet_group)
        else:
            self.init_resnet_block = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        skip_connect_dims = [] # keep tracking of skip connection dimensions
        upsample_combiner_dims = [] # keep track of dimensions for upsample feature combiner
        
        # down sampling part
        for ind, ((dim_in, dim_out), groups, layer_num_res_blocks, layer_self_attn) in enumerate(
            zip(in_out, resnet_groups, num_resnet_blocks, self_attn)):
            is_first = ind == 0
            is_last = ind >= (num_stages - 1)
            layer_cond_dim = cond_dim if not is_first else dim_in

            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))
            
            self.downs.append(nn.ModuleList([
                downsample_class(dim_in, dim_out=dim_out) if memory_efficient else None,
                resnet_block(dim_layer, dim_layer, time_cond_dim=time_cond_dim, groups=groups),
                nn.ModuleList([resnet_block(dim_layer, dim_layer, cond_dim=layer_cond_dim, time_cond_dim=time_cond_dim,
                                            groups=groups) for _ in range(layer_num_res_blocks)]),
                attention,
                downsample_class(dim_layer, dim_out=dim_out) if not is_last and not memory_efficient else conv_nd(unet_dim, dim_layer, dim_out, 1)
            ]))

        # middle part
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim, groups=resnet_groups[-1])
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim, groups=resnet_groups[-1])

        # up sampling part
        for ind, ((dim_in, dim_out), groups, layer_num_res_blocks, layer_self_attn) in enumerate(
            zip(*map(reversed, (in_out, resnet_groups, num_resnet_blocks, self_attn)))):

            is_last = ind == (len(in_out) - 1)
            layer_cond_dim = cond_dim if not is_last else None

            skip_connect_dim = skip_connect_dims.pop()

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))

            upsample_combiner_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_block(dim_out+skip_connect_dim, dim_out, cond_dim=layer_cond_dim, time_cond_dim=time_cond_dim, groups=groups),
                nn.ModuleList([resnet_block(dim_out+skip_connect_dim, dim_out, cond_dim=layer_cond_dim, time_cond_dim=time_cond_dim,
                                            groups=groups) for _ in range(layer_num_res_blocks)]),
                attention,
                upsample_class(dim_out, dim_in) if not is_last and not memory_efficient else nn.Identity()
            ]))
        
        # combine outputs from all upsample blocks for final resnet block
        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_features,
            dim_ins = upsample_combiner_dims,
            dim_outs = (dim,) * len(upsample_combiner_dims)
        )

        # final resnet block
        self.final_resnet_block = resnet_block(self.upsample_combiner.dim_out+dim, dim, time_cond_dim=time_cond_dim, groups=top_level_resnet_group)
        self.to_out = conv_nd(unet_dim, dim, self.out_channels, kernel_size=final_conv_kernel_size, padding=final_conv_kernel_size//2)
        
        # zero initialization
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)
    

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)
    

    def forward(
            self,
            x,
            time,
            *args,
            text_encodings=None,
            text_cond_drop_prob = 0.,
            self_cond=None
    ):
        batch_size = x.shape[0]
        
        # concat self conditioning
        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim=1)
        
        # initial convolution
        x = self.init_conv(x)
        r = x.clone() # for final residual

        # time conditioning
        time = time.type_as(x)
        time_hiddens = self.to_time_hiddens(time)
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditional dropout
        text_keep_mask = prob_mask_like((batch_size,), 1 - text_cond_drop_prob, device=x.device)
        text_keep_mask = rearrange(text_keep_mask, "B -> B 1 1")

        # deal with text encoding
        if text_encodings is not None and self.cond_on_text_encodings:
            assert text_encodings.shape[0] == batch_size, "Batch size of text encodings {} in the unet does not match with {}.".format(text_encodings.shape[0], batch_size)
            assert self.text_embed_dim == text_encodings.shape[-1], "Text encodings have a dimension of {}, but the unet was created with text_embed_dim of {}.".format(text_encodings.shape[-1], self.text_embed_dim)

            text_mask = torch.any(text_encodings!=0., dim=-1)[:, :self.max_text_len]
            text_tokens = self.text_to_cond(text_encodings)[:, :self.max_text_len]
            
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder)) # along 2 dims
                text_mask = F.pad(text_mask, (0, remainder), value=False) # along 1 dim
            text_mask = rearrange(text_mask, "B N -> B N 1")

            assert text_mask.shape[0] == text_keep_mask.shape[0], "Shape of text_mask is {}, text_keep_mask is {}, and text encoding is {}.".format(text_mask.shape, text_keep_mask.shape, text_encodings.shape)
            text_keep_mask = text_mask & text_keep_mask

            text_tokens = torch.where(
                text_keep_mask,
                text_tokens,
                null_text_embed
            )

        # main conditioning tokens (c)
        c = time_tokens

        # text conditioning tokens (mid_c)
        # to save on compute, only do cross attention based conditioning on the inner most layers of the Unet
        mid_c = c if text_tokens is None else torch.cat((c, text_tokens), dim=-2)

        # normalize conditioning tokens
        c = self.norm_cond(c)
        mid_c = self.norm_cond(mid_c)

        # layers
        if self.init_resnet_block is not None:
            x = self.init_resnet_block(x, t)
        
        down_hiddens = []
        up_hiddens = []
        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in self.downs:
            if pre_downsample is not None:
                x = pre_downsample(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                down_hiddens.append(x.contiguous())
            x = attn(x)
            down_hiddens.append(x.contiguous())
            if post_downsample is not None:
                x = post_downsample

        x = self.mid_block1(x, t, mid_c)
        if self.mid_attn is not None:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, mid_c)

        connect_skip = lambda feature: torch.cat((feature, down_hiddens.pop() * self.skip_connect_scale), dim=1)
        for init_block, resnet_blocks, attn, upsample in self.ups:
            x = connect_skip(x)
            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)
            x = attn(x)
            up_hiddens.append(x.contiguous())
            x = upsample(x)
        
        x = self.upsample_combiner(x, up_hiddens)
        x = torch.cat((x, r), dim=1)
        x = self.final_resnet_block(x, t)
        return self.to_out(x)
        


        if context is not None or context_cross is not None:
            raise NotImplementedError("Conditioning should be implemented additionally.")

        if self.self_cond:
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

