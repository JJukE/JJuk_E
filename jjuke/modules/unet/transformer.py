from typing import Optional

import numpy as np
import torch
from torch import nn
from einops import rearrange


class SpatialTransformer(nn.Module):
    """ Transformer for Latent Diffusion U-Net """
    def __init__(self, channels: int, n_head: int, n_layer: int, d_cond: int):
        """ init function
        
        Args:
            channels (int): number of channels in the feature map
            n_head (int): number of attention head
            n_layer (int): number of transformer layer
            d_cond (int): size of the conditional embedding
        """
        super().__init__()
        
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0) # 1 by 1 Conv
        
        # transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_head, channels // n_head, d_cond=d_cond) for _ in range(n_layer)]
        )
        
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """ forward function

        Args:
            x (torch.Tensor): feature map of shape (batch_size, channels, height, width)
            cond (torch.Tensor): conditional embedding of shape (batch_size, num_cond, dim_cond)
        """
        b, c, h, w = x.shape # (batch_size, channels, height, width)
        x_in = x # for residual connection
        
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        
        for block in self.transformer_blocks:
            x = block(x, cond)
        
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        
        x = x + x_in # add residual
        return x

    
class BasicTransformerBlock(nn.Module):
    """ Transformer layer """
    def __init__(self, d_emb: int, n_head: int, d_head: int, d_cond: int):
        """ init function

        Args:
            d_emb (int): input embedding size
            n_head (int): number of attention head
            d_head (int): size of attention head
            d_cond (int): size of the conditional embeddings. 0 means no conditioning.
        """
        super().__init__()
        
        self.self_attn = CrossAttention(d_emb, d_emb, n_head, d_head) # self-attention
        self.norm1 = nn.LayerNorm(d_emb)

        self.cross_attn = CrossAttention(d_emb, d_cond, n_head, d_head) # cross-attention
        self.norm2 = nn.LayerNorm(d_emb)
        
        self.ffn = FeedForward(d_emb) # feed-forward network
        self.norm3 = nn.LayerNorm(d_emb)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """ forward function

        Args:
            x (torch.Tensor): input embeddings of shape (batch_size, height * width, d_emb)
            cond (torch.Tensor): conditional embeddings of shape (batch_size, n_cond, d_cond)
        """
        x = self.self_attn(self.norm1(x)) + x # self-attention
        x = self.cross_attn(self.norm2(x), cond=cond) + x # cross-attention with conditioning
        x = self.ffn(self.norm3(x)) + x # feed-forward network
        return x

class CrossAttention(nn.Module):
    """ Self-attention or cross-attention layer """
    def __init__(self, d_emb: int, d_cond: int, n_head: int, d_head: int,
                    is_inplace: bool = True, use_flash_attention: bool = False):
        """ init function

        Args:
            d_emb (int): input embedding size
            d_cond (int): conditional embedding size
            n_head (int): number of attention head
            d_head (int): size of attention head
            is_inplace (bool, optional): specifies whether to perform the attention
                                         softmax computation inplace to save memory.
                                         Defaults to True.
            use_flash_attention (bool, optional): whether to use flash attention.
                                                  Defaults to False.
        """
        super().__init__()

        self.use_flash_attention = use_flash_attention
        self.is_inplace = is_inplace
        self.n_head = n_head
        self.d_head = d_head

        self.scale = d_head ** -0.5 # attention scaling factor

        # query, key and value mapping
        d_attn = d_head * n_head
        self.to_q = nn.Linear(d_emb, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond if d_cond != 0 else d_emb, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond if d_cond != 0 else d_emb, d_attn, bias=False)

        # final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_emb))

        # use flash attention if it is installed and use_flash_attention == True
        # installation : git clone https://github.com/HazyResearch/flash-attention
        # and run "python setup.py install"
        try:
            from flash_attn.flash_attention import FlashAttention
            self.flash = FalshAttention()
            self.flash.softmax_scale = self.scale # scale for scaled dot-product attention
        except ImportError:
            self.flash = None
    
    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ Flash Attention

        Flash attention works for head sizes 32, 64, and 128.

        Args:
            q (torch.Tensor): Query vectors before splitting heads of shape (batch_size, seq_query, d_attn).
            k (torch.Tensor): Key vectors before splitting heads of shape (batch_size, seq_key, d_attn).
            v (torch.Tensor): Value vectors before splitting heads of shape (batch_size, seq_value, d_attn).
        """
        batch_size, seq_len, _ = q.shape # (batch_size, width * height, n_head * d_head)
        
        qkv = torch.stack((q, k, v), dim=2) # (batch_size, seq_len, 3, n_head * d_head)
        qkv = rearrange(qkv, "b l three (n d) -> b l three n d", n=self.n_head, d=self.d_head) # (batch_size, seq_len, 3, n_head, d_head)
        
        # flash attention works for head sizes 32, 64 and 128
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError("Head size {} is too large for Flash Attention.".format(self.d_head))
        
        # pad the heads
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_head, pad)), dim=-1)
        
        # compute attention
        out, _ = self.flash(qkv) # (batch_size, seq_len, n_head, d_padded)
        out = out[:, :, :, :self.d_head] # truncate the extra head size
        out = rearrange(out, "b s n d -> b s (n d)")
        out = self.to_out(out) # (batch_size, height * width, d_emb)
        return out
    
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ Normal Attention

        Args:
            q (torch.Tensor): Query vectors before splitting heads of shape (batch_size, seq_query, d_attn).
            k (torch.Tensor): Key vectors before splitting heads of shape (batch_size, seq_key, d_attn).
            v (torch.Tensor): Value vectors before splitting heads of shape (batch_size, seq_value, d_attn).
        """
        # split
        q = rearrange(q, "b s (n d) -> b s n d", n=self.n_head, d=self.d_head) # (batch_size(b), seq_len of query, n_head(h), d_head(d))
        k = rearrange(k, "b s (n d) -> b s n d", n=self.n_head, d=self.d_head) # (batch_size(b), seq_len of key, n_head(h), d_head(d))
        v = rearrange(v, "b s (n d) -> b s n d", n=self.n_head, d=self.d_head)
        
        # compute attention
        attn = torch.einsum("b i h d, b j h d -> b h i j", q, k) # batch(K^T): (b d h j) -> batch matrix multiplication
        
        # compute softmax
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[:half] = attn[:half].softmax(dim=-1)
            attn[half:] = attn[half:].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        
        # compute attention output
        out = torch.einsum("bhij,bjhd->bihd", attn, v)
        out = rearrange(out, "b i h d -> b i (h d)")
        return self.to_out(out)
        
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """ forward function

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, height * width, d_emb)
            cond (Optional[torch.Tensor], optional): conditional embeddings of shape (batch_size, n_cond, d_cond).
                                                     Defaults to None.
        """
        if cond == None:
            cond = x # self-attention
        
        # get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        
        # use flash attention if it's available and the head size is less than or equal to 128
        if self.use_flash_attention and self.flash is not None and cond is None and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        else: # normal attention
            return self.normal_attention(q, k, v)

class FeedForward(nn.Module):
    """ Feed-Forward Network (FFN)"""
    def __init__(self, d_emb: int, d_multi: int = 4):
        """ init function

        Args:
            d_emb (int): input embedding size
            d_multi (int, optional): hidden layer size. Defaults to 4.
        """
        super().__init__()
        
        self.net = nn.Sequential(
            GeGLU(d_emb, d_emb * d_multi),
            nn.Dropout(0.),
            nn.Linear(d_emb * d_multi, d_emb)
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

class GeGLU(nn.Module):
    """ GeGLU activation function """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        
        self.proj = nn.Linear(d_in, d_out * 2) # combined linear projection ($xW + b$ and $xV + c$)
    
    def forward(self, x: torch.Tensor):
        print(x.shape) # TODO: shape of x? -> chunk to rearrange: x, gate = rearrange(self.proj(x), "")
        x, gate = self.proj(x).chunk(2, dim=-1)