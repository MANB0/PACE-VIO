""" Twins
A PyTorch impl of : `Twins: Revisiting the Design of Spatial Attention in Vision Transformers`
    - https://arxiv.org/pdf/2104.13840.pdf
Code/weights from https://github.com/Meituan-AutoML/Twins, original copyright/license info below
"""
# --------------------------------------------------------
# Twins
# Copyright (c) 2021 Meituan
# Licensed under The Apache 2.0 License [see LICENSE for details]
# Written by Xinjie Li, Xiangxiang Chu
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LinearPositionEmbeddingSine
from .utils import coords_grid
from .Twins.svt_large import Mlp



class LocallyGroupedAttnRPEContext(nn.Module):
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1, vert_c_dim=0):
        assert ws != 1
        super(LocallyGroupedAttnRPEContext, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.vert_c_dim = vert_c_dim

        self.context_proj = nn.Linear(256, vert_c_dim)
        # context are not added to value
        self.q = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: tuple[int, int], context: torch.Tensor):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        C_qk = C+self.vert_c_dim

        context = context.repeat(B//context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H*W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)

        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x     = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk  = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x    = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        x_qk = x_qk.reshape(B, _h, self.ws, _w, self.ws, C_qk).transpose(2, 3)


        coords = coords_grid(B, self.ws, self.ws, x.device, x.dtype)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk).view(B, self.ws, self.ws, C_qk)   
        # coords_enc:   B, ws, ws, C
        # x:            B, _h, _w, self.ws, self.ws, C
        x_qk = x_qk + coords_enc[:, None, None, :, :, :]


        # helper dims
        ntok     = self.ws * self.ws
        head_dim = C // self.num_heads

        q = self.q(x_qk).reshape(B, _h, _w, self.ws, self.ws, self.num_heads, head_dim)
        k = self.k(x_qk).reshape(B, _h, _w, self.ws, self.ws, self.num_heads, head_dim)
        v = self.v(x).reshape(B, _h, _w, self.ws, self.ws, self.num_heads, head_dim)

        q = q.permute(0, 1, 2, 5, 3, 4, 6)
        k = k.permute(0, 1, 2, 5, 3, 4, 6)
        v = v.permute(0, 1, 2, 5, 3, 4, 6)

        B2 = B * _h * _w
        q_flat = q.reshape(B2, self.num_heads, ntok, head_dim)
        k_flat = k.reshape(B2, self.num_heads, ntok, head_dim)
        v_flat = v.reshape(B2, self.num_heads, ntok, head_dim)

        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()

        attn_flat = F.scaled_dot_product_attention(
            q_flat, k_flat, v_flat,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )

        # reshape back to windowed layout (split ntok into ws, ws)
        attn = attn_flat.reshape(B, _h, _w, self.num_heads, self.ws, self.ws, head_dim)
        attn = attn.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, _h * self.ws, _w * self.ws, C)

        x = attn[:, :H, :W, :]

        # final project
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttnRPEContext(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1, vert_c_dim=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.vert_c_dim   = vert_c_dim
        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q            = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.k            = nn.Linear(dim, dim, bias=True)
        self.v            = nn.Linear(dim, dim, bias=True)
        self.attn_drop    = nn.Dropout(attn_drop)
        self.proj         = nn.Linear(dim, dim)
        self.proj_drop    = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.sr_key   = nn.Conv2d(dim+vert_c_dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.sr_value = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm     = nn.LayerNorm(dim)

    def forward(self, x, size: tuple[int, int], context: torch.Tensor):
        B, N, C = x.shape
        C_qk = C + self.vert_c_dim
        H, W = size
        context = context.repeat(B//context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H*W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)
        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        _, Hp, Wp, _ = x.shape
        padded_size = (Hp, Wp)
        padded_N = Hp*Wp
        x = x.view(B, -1, C)
        x_qk = x_qk.view(B, -1, C_qk)

        coords = coords_grid(B, *padded_size, x.device, x.dtype)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)   
        # coords_enc:   B, Hp*Wp, C
        # x:            B, Hp*Wp, C
        q = self.q(x_qk + coords_enc).reshape(B, padded_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
        x_qk = x_qk.permute(0, 2, 1).reshape(B, C_qk, *padded_size)
        x = self.sr_value(x).reshape(B, C, -1).permute(0, 2, 1)
        x_qk = self.sr_key(x_qk).reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        x_qk = self.norm(x_qk)

        coords = coords_grid(B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio, x.device, x.dtype)
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        
        # align the coordinate of local and global
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = self.k(x_qk + coords_enc).reshape(B, (padded_size[0] // self.sr_ratio)*(padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, (padded_size[0] // self.sr_ratio)*(padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p,
            is_causal=False
        )
        
        x = x.transpose(1, 2).reshape(B, Hp, Wp, C)
        # if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :]
        # endif

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        mlp_ratio: float=4.,
        drop: float=0.,
        attn_drop: float=0.,
        sr_ratio=1,
        ws: int=1,
        vert_c_dim=0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        if ws == 1:
            self.attn = GlobalSubSampleAttnRPEContext(dim, num_heads, attn_drop, drop, sr_ratio, vert_c_dim)
        else:
            self.attn = LocallyGroupedAttnRPEContext(dim, num_heads, attn_drop, drop, ws, vert_c_dim)
        
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=(drop, drop))

    def forward(self, x: torch.Tensor, size: tuple[int, int], context=None):
        with torch.cuda.nvtx.range(f"Attention [{self.attn.__class__.__name__}]"):
            x = x + self.drop_path(self.attn(self.norm1(x), size, context))
        
        with torch.cuda.nvtx.range("MLP"):
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
