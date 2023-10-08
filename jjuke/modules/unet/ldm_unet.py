import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from unet.transformer import SpatialTransformer


class UNetModel(nn.Module):
    """ U-Net for Latent Diffusion"""
    def __init__(
            self, *,
            in_channels: int,
            out_channels: int, # TODO: 같아야 하는 거 아닌가?
            channels: int,
            n_res_blocks: int,
            attention_levels: List[int],
            channel_multipliers: List[int],
            n_heads: int,
            tf_layers: int = 1,
            d_cond: int = 768):
        """ init funciton

        Args:
            in_channels (int): Number of channels in the input feature map
            out_channels (int): Number of channels in the output feature map
            channels (int): Base channel count for the model
            n_res_blocks (int): Number of residual blocks at each level
            attention_levels (List[int]): Levels at which attention should be applied
            channel_multipliers (List[int]): Multiplicative factors for number of channels for each level
            n_heads (int): Number of attention heads in the transformers
            tf_layers (int, optional): Number of transformer layers in the transformers. Defaults to 1.
            d_cond (int, optional): Size of conditional embedding in the transformers. Defaults to 768.
        """        

        super().__init__()

        self.channels = channels

        levels = len(channel_multipliers) # numbmer of levels
        d_time_emb = channels * 4 # size of time embdding
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb)
        )

        # first(input) half of the U-Net
        self.input_blocks = nn.ModuleList() # input half of the U-Net
        self.input_blocks.append(
            TimestepEmbedSequential(nn.Conv2d(in_channels, channels, 3, padding=1))
        )
        input_block_channels = [channels] # num of channels at each block
        channels_list = [channels * m for m in channel_multipliers] # num of channels at each level
        for i in range(levels):
            # add residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block: num of channels in prev level -> num of channels in current level 
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                
                # compose the layers of input half of the U-Net
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            # down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
        
        # the middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, d_time_emb),
        )

        # second(output) half of the U-Net
        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            # add residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block: num of channels in prev level + skip connections from input
                # -> num of channels in current level
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]

                # add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                # Up-sample at every level after last residual block except the last one
                if i != 0 and j == n_res_blocks: # i==0: last
                    layers.append(UpSample(channels))
                # add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        # final normalization and 3 by 3 conv
        self.out = nn.Sequential(
            GroupNorm32(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1)
        )
    
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """ Create sinusoidal time step embeddings

        Args:
            time_steps (torch.Tensor): Time steps of shape (batch_size).
            max_period (int): Maximum period to control minimum frequency of embeddings.
        Returns:
            Time embedding
        """        
        # half of the channels is sine, the other half is cosine
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device) # 1 / 10000^{2i/c}
        args = time_steps[:,None].float() * frequencies[None] # t / 10000^{2i/c}
        time_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # cosine and sine
        return time_emb
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor = None):
        """ Forward function

        Args:
            x (torch.Tensor): Input feature map of shape (batch_size, channels, width, height)
            time_steps (torch.Tensor): time steps of shape (batch_size)
            cond (torch.Tensor): conditioning of shape (batch_size, n_cond, d_cond)
        """
        x_input_block = [] # store input half outputs for skip connections
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        x = self.middle_block(x, t_emb, cond)
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)
        x = self.out(x)
        return x


class TimestepEmbedSequential(nn.Sequential):
    """ Sequential block for modules with different inputs
    This sequential module can compose of different modules such as ResBlock, nn.Conv, and SpatialTransformer
    and calls them with corresponding inputs
    """
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """ Up-sampling layer """
    def __init__(self, channels: int):
        """ Init function

        Args:
            channels (int): number of channels
        """
        super().__init__()
        # 3 by 3 conv mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """ Forward function

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, channels, height, width)
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest") # up-sample by a factor of 2
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    """ Down-sampling layer """
    def __init__(self, channels: int):
        """ Init function

        Args:
            channels (int): number of channels
        """        
        super().__init__()
        # 3 by 3 conv with stride length of 2 to down-sample
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """ Forward function

        Args:
            x (torch.Tensor): Input feature map with shape (batch_size, channels, height, width)
        """
        x = self.conv(x) 
        return x


class ResBlock(nn.Module):
    """ ResNet Block """
    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """ init funciton

        Args:
            channels (int): Number of input channels
            d_t_emb (int): Size of timestep embeddings
            out_channels (_type_, optional): Number of output channels. Defaults to None.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            GroupNorm32(out_channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # Final convolution layer
        self.out_layers = nn.Sequential(
            GroupNorm32(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Time step embeddings
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # Add time step embeddings
        h = h + t_emb[:, :, None, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """ Group normalization with float32 casting """
    def __init__(self, num_channels):
        super().__init__(32, num_channels) # 32 for num_groups of nn.GroupNorm

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype) # casting input as float32


def _test_time_embeddings(): # TODO: check how this function works
    """ Test sinusoidal time step embeddings """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    m = UNetModel(in_channels=1, out_channels=1, channels=320, n_res_blocks=1, attention_levels=[],
                  channel_multipliers=[], n_heads=1, tf_layers=1, d_cond=1)
    te = m.time_step_embedding(torch.arange(0, 1000))
    plt.plot(np.arange(1000), te[:, [50, 100, 190, 260]].numpy())
    plt.legend(["dim %d" % p for p in [50, 100, 190, 260]])
    plt.title("Time embeddings")
    plt.show()
