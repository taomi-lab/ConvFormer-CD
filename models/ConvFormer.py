# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/2 H3:11
@Author  : LMT
"""
import math
import torch
import torch.nn as nn
from models.model_utils import *
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models.Exchange_Module import ChannelExchange, SpatialExchange
from models.Difference_Integration_Module import DIM
from models.Local_Enhancement_Module import LFE


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if self.attn_drop:
            m_r = torch.ones_like(attn) * self.attn_drop
            attn = attn + torch.bernoulli(m_r) * -1e12

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class TemporalAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        self.proj1 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop2 = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, mask=None):
        B_, N, C = x1.shape
        kv1 = self.kv1(x1).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # make torchscript happy (cannot use tensor as tuple)
        kv2 = self.kv2(x2).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]
        q = self.q(torch.abs(x2 - x1)).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # x2
        q = q * self.scale
        attn = (F.normalize(q) @ F.normalize(k2).transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if self.attn_drop:
            m_r = torch.ones_like(attn) * self.attn_drop
            attn = attn + torch.bernoulli(m_r) * -1e12
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x1 = torch.sub((attn @ v2), q).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj1(x1)
        x1 = self.proj_drop1(x1)

        # x1
        q = q * self.scale
        attn = (F.normalize(q) @ F.normalize(k1).transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if self.attn_drop:
            m_r = torch.ones_like(attn) * self.attn_drop
            attn = attn + torch.bernoulli(m_r) * -1e12
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x2 = torch.sub((attn @ v1), q).transpose(1, 2).reshape(B_, N, C)
        x2 = self.proj2(x2)
        x2 = self.proj_drop2(x2)

        return x1, x2

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class ConvFormerDecoderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.Conv_branch = LFE(self.dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn1(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x_conv = blc_to_bchw(shortcut, self.input_resolution)
        # FFN
        x = shortcut + self.drop_path(x) + bchw_to_blc(x_conv)
        x = x + self.drop_path(self.mlp1(self.norm1(x)))

        return x


class BTBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Exchange_type=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.Exchange_type = Exchange_type
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = TemporalAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        if self.Exchange_type == "ChannelExchange":
            self.Exchange = ChannelExchange()
        if self.Exchange_type == "SpatialExchange":
            self.Exchange = SpatialExchange()
        self.register_buffer("attn_mask", attn_mask)
        self.Conv_branch1 = LFE(self.dim)
        self.Conv_branch2 = LFE(self.dim)

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L, C = x1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut1 = x1
        x1 = self.norm1(x1)
        x1 = x1.view(B, H, W, C)
        shortcut2 = x2
        x2 = self.norm1(x2)
        x2 = x2.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        # partition windows
        x_windows1 = window_partition(shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
        x_windows2 = window_partition(shifted_x2, self.window_size)
        x_windows1 = x_windows1.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows2 = x_windows2.view(-1, self.window_size * self.window_size, C)

        attn_windows1, attn_windows2 = self.attn(x_windows1, x_windows2,
                                                 mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows1 = attn_windows1.view(-1, self.window_size, self.window_size, C)
        attn_windows2 = attn_windows2.view(-1, self.window_size, self.window_size, C)
        shifted_x1 = window_reverse(attn_windows1, self.window_size, H, W)  # B H' W' C
        shifted_x2 = window_reverse(attn_windows2, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(shifted_x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x1
            x2 = shifted_x2
        x1 = x1.view(B, H * W, C)
        x2 = x2.view(B, H * W, C)

        x1_conv = blc_to_bchw(shortcut1, self.input_resolution)
        x2_conv = blc_to_bchw(shortcut2, self.input_resolution)

        if self.Exchange_type == "ChannelExchange":
            x1_conv, x2_conv = self.Exchange(x1_conv, x2_conv)
        if self.Exchange_type == "SpatialExchange":
            x1_conv, x2_conv = self.Exchange(x1_conv, x2_conv)

        x1_conv = self.Conv_branch1(x1_conv)
        x2_conv = self.Conv_branch2(x2_conv)

        # FFN
        x1 = shortcut1 + self.drop_path(x1) + bchw_to_blc(x1_conv)
        x2 = shortcut2 + self.drop_path(x2) + bchw_to_blc(x2_conv)
        x1 = x1 + self.drop_path(self.mlp1(self.norm2(x1)))
        x2 = x2 + self.drop_path(self.mlp2(self.norm2(x2)))

        return x1, x2

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp1
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class STBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.Conv_branch1 = LFE(self.dim)
        self.Conv_branch2 = LFE(self.dim)

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L, C = x1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut1 = x1
        shortcut2 = x2
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        x1 = x1.view(B, H, W, C)
        x2 = x2.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        # partition windows
        x_windows1 = window_partition(shifted_x1, self.window_size)
        x_windows2 = window_partition(shifted_x2, self.window_size)
        x_windows1 = x_windows1.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows2 = x_windows2.view(-1, self.window_size * self.window_size, C)

        attn_windows1 = self.attn1(x_windows1, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        attn_windows2 = self.attn2(x_windows2, mask=self.attn_mask)

        # merge windows
        attn_windows1 = attn_windows1.view(-1, self.window_size, self.window_size, C)
        shifted_x1 = window_reverse(attn_windows1, self.window_size, H, W)  # B H' W' C
        attn_windows2 = attn_windows2.view(-1, self.window_size, self.window_size, C)
        shifted_x2 = window_reverse(attn_windows2, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x2 = torch.roll(shifted_x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x1
            x2 = shifted_x2
        x1 = x1.view(B, H * W, C)
        x2 = x2.view(B, H * W, C)

        x1_conv = self.Conv_branch1(blc_to_bchw(shortcut1, self.input_resolution))
        x2_conv = self.Conv_branch2(blc_to_bchw(shortcut2, self.input_resolution))

        # FFN
        x1 = shortcut1 + self.drop_path(x1) + bchw_to_blc(x1_conv)
        x2 = shortcut2 + self.drop_path(x2) + bchw_to_blc(x2_conv)
        x1 = x1 + self.drop_path(self.mlp1(self.norm2(x1)))
        x2 = x2 + self.drop_path(self.mlp2(self.norm2(x2)))

        return x1, x2


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        # self.input_resolution = (input_resolution[0]*2,)*2

        self.dim = dim
        self.expand = nn.Linear(dim, dim * 2, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        # H = W = H / 2
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)

        x = self.norm(x)

        return x


class ConvFormerEncoderLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=None, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 act_layer=None, Exchange_type=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2:
                self.blocks.append(BTBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                                   i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i]
                                           if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer,
                                           Exchange_type=Exchange_type))
            else:
                self.blocks.append(STBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i]
                                           if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer, ))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x1, x2):
        for blk in self.blocks:
            x1, x2 = blk(x1, x2)
        if self.downsample is not None:
            x1 = self.downsample(x1)
            x2 = self.downsample(x2)
        return x1, x2

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class ConvFormerDecoder(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=None, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConvFormerDecoderBlock(dim=dim, input_resolution=input_resolution,
                                                      num_heads=num_heads, window_size=window_size,
                                                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                      mlp_ratio=mlp_ratio,
                                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                      drop=drop, attn_drop=attn_drop,
                                                      drop_path=drop_path[i]
                                                      if isinstance(drop_path, list) else drop_path,
                                                      norm_layer=norm_layer, ))

        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Fusion_module(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(Fusion_module, self).__init__()
        # channel attention �)H,W:1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv1d(channel * 3, channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.conv2 = nn.Conv1d(channel * 3, channel, kernel_size=spatial_kernel, padding=spatial_kernel // 2,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        xd = torch.sub(x1, x2)
        x1_pool = torch.cat([torch.max(x1, dim=1, keepdim=True)[0], torch.mean(x1, dim=1, keepdim=True)], dim=1)
        x2_pool = torch.cat([torch.max(x2, dim=1, keepdim=True)[0], torch.mean(x2, dim=1, keepdim=True)], dim=1)
        xd_pool = torch.cat([torch.max(xd, dim=1, keepdim=True)[0], torch.mean(xd, dim=1, keepdim=True)], dim=1)

        x1_spatial_out = self.sigmoid(self.conv(x1_pool)) * x1
        x2_spatial_out = self.sigmoid(self.conv(x2_pool)) * x2
        xd_spatial_out = self.sigmoid(self.conv(xd_pool)) * xd
        x_all = torch.cat([x1_spatial_out, x2_spatial_out, xd_spatial_out], dim=1)

        x_all_pool = self.sigmoid(self.mlp(self.max_pool(x_all)) + self.mlp(self.avg_pool(x_all)))

        x_all = self.conv2(x_all) * x_all_pool

        return x_all


class ConvFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2, embed_dim=48,
                 encoder_depths=[2, 2, 6, 2], decoder_depths=[6, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False):
        super().__init__()

        print("SCUNet initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}"
              .format(encoder_depths, decoder_depths, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(encoder_depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.linear_projection = nn.Linear(embed_dim * 8 * 2, embed_dim * 8, bias=False)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_attn_drop_rates = [0.1, 0.05, 0.01, 0.01]
        # build encoder encoder_layers
        self.encoder_layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]  # stochastic depth decay rule
        Exchange_types = [None, "SpatialExchange", "ChannelExchange", "ChannelExchange"]
        for i_layer in range(self.num_layers):
            layer = ConvFormerEncoderLayer(dim=int(embed_dim * 2 ** i_layer),
                                           input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                             patches_resolution[1] // (2 ** i_layer)),
                                           depth=encoder_depths[i_layer],
                                           num_heads=num_heads[i_layer],
                                           window_size=window_size,
                                           mlp_ratio=self.mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop_rate, attn_drop=encoder_attn_drop_rates[i_layer],
                                           drop_path=dpr[sum(encoder_depths[:i_layer]):
                                                         sum(encoder_depths[:i_layer + 1])],
                                           norm_layer=norm_layer,
                                           downsample=PatchMerging if i_layer != 3 else None,
                                           use_checkpoint=use_checkpoint,
                                           Exchange_type=Exchange_types[i_layer])
            self.encoder_layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build decoder encoder_layers
        self.decoder_layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]  # stochastic depth decay rule
        decoder_layers = len(decoder_depths)
        encoder_attn_drop_rates = [0.01, 0.05, 0.1]
        for i_layer in range(decoder_layers):
            layer = ConvFormerDecoder(dim=int(embed_dim * 2 ** (decoder_layers - i_layer)),
                                      input_resolution=(
                                          patches_resolution[0] // (2 ** (decoder_layers - i_layer)),
                                          patches_resolution[1] // (2 ** (decoder_layers - i_layer))),
                                      depth=decoder_depths[i_layer],
                                      num_heads=num_heads[i_layer],
                                      window_size=window_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=encoder_attn_drop_rates[i_layer],
                                      drop_path=dpr[sum(decoder_depths[:i_layer]):
                                                    sum(decoder_depths[:i_layer + 1])],
                                      norm_layer=norm_layer,
                                      upsample=PatchExpand,
                                      use_checkpoint=use_checkpoint)
            self.decoder_layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                      dim_scale=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)
        self.fusion = nn.ModuleList()
        for i_layer in range(decoder_layers):
            self.fusion.append(DIM(embed_dim * 2 ** (decoder_layers - i_layer - 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features_encoder(self, x1, x2):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed
        x1_downsample = []
        x2_downsample = []
        x1_downsample.append(x1)
        x2_downsample.append(x2)
        for i, layer in enumerate(self.encoder_layers):
            x1, x2 = layer(x1, x2)
            x1_downsample.append(x1)
            x2_downsample.append(x2)
        return x1_downsample, x2_downsample

    def forward_features_decoder(self, x1_downsample, x2_downsample):
        x1 = x1_downsample[-1] + x1_downsample[-2]
        x2 = x2_downsample[-1] + x2_downsample[-2]
        x = x1 + x2
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            x1 = blc_to_bchw(x1_downsample[2 - i], list(np.array(self.patches_resolution) // (2 ** (2 - i))))
            x2 = blc_to_bchw(x2_downsample[2 - i], list(np.array(self.patches_resolution) // (2 ** (2 - i))))
            x += bchw_to_blc(self.fusion[i](x1, x2))
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)

        return x

    def forward(self, x1, x2):
        x1_downsample, x2_downsample = self.forward_features_encoder(x1, x2)
        x = self.forward_features_decoder(x1_downsample, x2_downsample)
        x = self.up_x4(x)
        return x

def save_json(model_dict, file_name):
    import json
    for key in model_dict:
        model_dict[key] = None
    with open(file_name, 'w') as f:
        json.dump(model_dict, f, indent=4)


if __name__ == '__main__':
    model_path = "/data/lmt/projects/SCUNet/outputs/SCUNetSingle_48_LEVIR_11-17-10-19/best_model.pth"
    model = ConvFormer(img_size=224)
    pre_model = torch.load(model_path, map_location="cpu")["model_G_state_dict"]
    from copy import deepcopy
    now_model = deepcopy(model.state_dict())
    pre_model_keys = list(pre_model.keys())
    now_model_keys = list(now_model.keys())
    for i in range(len(pre_model_keys)):
        now_model[now_model_keys[i]] = pre_model[pre_model_keys[i]]


    # save_json(pre_model, "pre_dict.json")
    # save_json(now_model, "now_dict.json")
    # pre_model_keys = list(pre_model.keys())
    # now_model_keys = list(now_model.keys())
    # with open("contrast.txt", 'w') as f:
    #     for i in range(len(pre_model_keys)):
    #         if len(pre_model_keys[i].split(".")) != len(now_model_keys[i].split(".")):
    #             f.write(f"{pre_model_keys[i]}, {now_model_keys[i]} \n")

    print(len(pre_model.keys()), len(now_model.keys()))
    #
    model.load_state_dict(now_model)
    img = torch.randn(2, 3, 224, 224)
    x = model(img, img)
    print(x.shape)
