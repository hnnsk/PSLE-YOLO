#----------swintf----C3STR---------------------------------
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from ultralytics.nn.modules.block import C3

__all__ = (
    "C3STR",
    "PatchMerging",
    "MDCR",
    "PPA",
    "FPIC",
    "PIC",
    "Feature_selection",
    "FEM",
    "FFM_Concat2",
    "FFM_Concat3",
    "SCAM",
    "DASI3",
    "DASI2",
    "DASI1",
    "DASI2h",
    "DASI3h",
    "SPD",
)

class SPD(nn.Module):
    def __init__(self,block_size=2):
        super(SPD,self).__init__()
        self.block_size = block_size

    def forward(self,x):
        N,C,H,W=x.size()
        block_size = self.block_size
        assert H%block_size ==0 and W%block_size ==0,\
            f"空间维度必须能被block_size整除。得到的H：{H}，W：{W}"

        x_reshaped = x.view(N, C, H//block_size, block_size, W//block_size, block_size)
        x_permuted = x_reshaped.permute(0,3,5,1,2,4).contiguous()
        out = x_permuted.view(N, C*block_size**2, H//block_size, W//block_size)

        return out


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2) for i in
                                      range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.normal_(self.relative_position_bias_table, std=.02)
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

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # print(attn.dtype, v.dtype)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            # print(attn.dtype, v.dtype)
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##SwinTransformerLayer--不包含patch-merging，仅为SwinTransformer中的一个stage，通过num-layers为2的倍数完成W-MHSA和SW-MHSA的组合使用
class SwinTransformerLayer(nn.Module):

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
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

        def window_partition(x, window_size):
            """
            Args:
                x: (B, H, W, C)
                window_size (int): window size
            Returns:
                windows: (num_windows*B, window_size, window_size, C)
            """
            B, H, W, C = x.shape
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows

        def window_reverse(windows, window_size, H, W):
            """
            Args:
                windows: (num_windows*B, window_size, window_size, C)
                window_size (int): Window size
                H (int): Height of image
                W (int): Width of image
            Returns:
                x: (B, H, W, C)
            """
            B = int(windows.shape[0] / (H * W / window_size / window_size))
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        def window_partition(x, window_size):
            """
            Args:
                x: (B, H, W, C)
                window_size (int): window size
            Returns:
                windows: (num_windows*B, window_size, window_size, C)
            """
            B, H, W, C = x.shape
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows

        def window_reverse(windows, window_size, H, W):
            """
            Args:
                windows: (num_windows*B, window_size, window_size, C)
                window_size (int): Window size
                H (int): Height of image
                W (int): Width of image
            Returns:
                x: (B, H, W, C)
            """
            B = int(windows.shape[0] / (H * W / window_size / window_size))
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x


class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

import torch
import torch.nn as nn
import math
import numpy as np


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, act=True):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, H, W = x.shape

        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C)

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
        x = x.permute(0,2,3,1)
        x = x.view(B, -1, 4 * C)  # B 4*C H/2*W/2

        x = self.norm(x)
        x = self.reduction(x)
        L = x.shape[1]

        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        x = x.reshape(B,H,W,C*2)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)

        return x



class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class MDCR(nn.Module):
    def __init__(self, c1, c2, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=c1//4,
            out_features=c2//4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups = 36
            )
        self.block2 = conv_block(
            in_features=c1//4,
            out_features=c2//4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=36
            )
        self.block3 = conv_block(
            in_features=c1//4,
            out_features=c2//4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=36
            )
        self.block4 = conv_block(
            in_features=c1//4,
            out_features=c2//4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=36
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=c2,
            out_features=c2,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)    H/P*W/P等价于把一个通道上的像素flatten，P*P等价于有C个channel

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)
        ##feature selection##
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform
        ##feature selection##
        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=False)
         self.c1 = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c2 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c3 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.bn1 = nn.BatchNorm2d(filters)
         self.drop = nn.Dropout2d(0.1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()
         self.MP = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.MP(x)
        return x


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))



class Patch_interaction(nn.Module):
    def __init__(self, output_dim, patch_size, act=True):
        super(Patch_interaction, self).__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)    H/P*W/P等价于把一个通道上的像素flatten，P*P等价于有C个channel

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)
        local_out = local_out.reshape(B, H//P, W//P, self.output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        return local_out




class FPIC(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2//4, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)     ##conv2d(1*1) backbone
        self.Focus = Focus(c2, c2)
        self.PI = Patch_interaction(c2, 2)
        self.conv2d = nn.Conv2d(c2//4, c2, 3, 2, 1, groups=g, dilation=d, bias=False)     ##conv2d(3*3)branch
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.Focus(x)
        x2 = self.conv2d(x)
        x3 = self.PI(x)
        x = x1 + x2 + x3
        x = self.bn(x)
        x = self.act(x)
        return x

class PIC(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.PI = Patch_interaction(c2, 2)
        self.conv2d = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)     ##conv2d(3*3)branch
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x1 = self.conv2d(x)
        x2 = self.PI(x)
        x = x1 + x2
        x = self.bn(x)
        x = self.act(x)
        return x


class Feature_selection(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.conv2d = nn.Conv2d(c2, c2, 3, 1, 1, groups=1, dilation=1, bias=False)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(c2, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(c2), requires_grad=True)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # l = len(x)
        # x_tensor = []
        # # if self.i ==10:
        # #     print('ok')
        # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if l==2:
        #     B0, C0, H0, W0 = x[0].shape
        #     B1, C1, H1, W1 = x[1].shape
        #     if self.i==12:   ##这里的序号要根据yaml中实际顺序来改
        #         x[1] = F.interpolate(x[1], size=(H0, W0), mode='bilinear', align_corners=False)
        #     else:
        #         x[0] = F.interpolate(x[0], size=(H1, W1), mode='bilinear', align_corners=False)
        #
        #     input = torch.cat((x[0], x[1]), dim=1)
        #
        # elif l==3:
        #     B, C, H, W = x[1].shape
        #     x[0] = F.interpolate(x[0], size=(H, W), mode='bilinear', align_corners=False)
        #     x[2] = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        #     input = torch.cat((x[0], x[1], x[2]),dim=1)

        out1 = self.conv(x)
        out1 = self.conv2d(out1)
        out2 = self.conv(x)
        out = out2 + out1
        # input = input.permute(0, 2, 3, 1)
        # B, H, W, C = input.shape   ##C=output_dim
        # input = input.reshape(B, -1, C)
        #
        # cos_sim = F.normalize(input, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        # mask = cos_sim.clamp(0, 1)
        # input = input * mask
        # input = input @ self.top_down_transform
        #
        # input = input.reshape(B, H, W, C)  # (B, H, W, C)
        # input = input.permute(0, 3, 1, 2)

        out = self.bn(out)
        output = self.act(out)

        return output


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FFM_Concat2(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1):
        super(FFM_Concat2, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = int(Channel1 + Channel2)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        w = self.w[:(C1 + C2)] # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x = [x1, x2]
        return torch.cat(x, self.d)

class FFM_Concat3(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1, Channel3 = 1):
        super(FFM_Concat3, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel3 = Channel3
        self.Channel_all = int(Channel1 + Channel2 + Channel3)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()
        N3, C3, H3, W3 = x[2].size()

        w = self.w[:(C1 + C2 + C3)]  # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:(C1 + C2)] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x3 = (weight[(C1 + C2):] * x[2].view(N3, H3, W3, C3)).view(N3, C3, H3, W3)
        x = [x1, x2, x3]
        return torch.cat(x, self.d)

class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class SCAM(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # avg max: [N, C, 1, 1]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y2:[N, 1, H, W]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat:[N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y

class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

class DASI3(nn.Module):
    def __init__(self, in_features_low, in_features, in_features_high,  out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)
         self.skips_2 = conv_block(in_features=in_features_high,
                                 out_features=out_features,
                                 kernel_size=(1, 1),
                                 padding=(0, 0),
                                 norm_type=None,
                                 activation=False)
         self.skips_3 = nn.Conv2d(in_features_low, out_features,
                                  kernel_size=3, stride=2, dilation=2, padding=2)
         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        if x[0] != None:
            x[0] = self.skips_3(x[0])
            x_low = torch.chunk(x[0], 4, dim=1)
        if x[2] != None:
            x[2] = self.skips_2(x[2])
            x[2] = F.interpolate(x[2], size=[x[1].size(2), x[1].size(3)], mode='bicubic', align_corners=True)
            x_high = torch.chunk(x[2], 4, dim=1)
        x_skip = self.skips(x[1])
        x[1] = self.skips(x[1])
        x_mid = torch.chunk(x[1], 4, dim=1)

        x0 = self.bag(x_low[0], x_high[0], x_mid[0])
        x1 = self.bag(x_low[1], x_high[1], x_mid[1])
        x2 = self.bag(x_low[2], x_high[2], x_mid[2])
        x3 = self.bag(x_low[3], x_high[3], x_mid[3])

        xx = torch.cat((x0, x1, x2, x3), dim=1)
        xx = self.tail_conv(xx)
        xx += x_skip
        xx = self.bns(xx)
        xx = self.relu(xx)

        return xx

class DASI3h(nn.Module):
    def __init__(self, in_features_low, in_features, in_features_high,  out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)
         self.skips_2 = conv_block(in_features=in_features_high,
                                 out_features=out_features,
                                 kernel_size=(1, 1),
                                 padding=(0, 0),
                                 norm_type=None,
                                 activation=False)
         self.skips_3 = nn.Conv2d(in_features_low, out_features,
                                  kernel_size=3, stride=2, dilation=2, padding=2)
         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        if x[0] != None:
            x[0] = self.skips_3(x[0])
            x_low = torch.chunk(x[0], 4, dim=1)
        if x[2] != None:
            x[2] = self.skips_2(x[2])
            x[2] = F.interpolate(x[2], size=[x[1].size(2), x[1].size(3)], mode='bicubic', align_corners=True)
            x_high = torch.chunk(x[2], 4, dim=1)
        x_skip = self.skips(x[1])
        x[1] = self.skips(x[1])
        x_mid = torch.chunk(x[1], 4, dim=1)

        x0 = self.bag(x_low[0], x_high[0], x_mid[0])
        x1 = self.bag(x_low[1], x_high[1], x_mid[1])
        x2 = self.bag(x_low[2], x_high[2], x_mid[2])
        x3 = self.bag(x_low[3], x_high[3], x_mid[3])

        xx = torch.cat((x0, x1, x2, x3), dim=1)
        xx = self.tail_conv(xx)
        xx += x_skip
        xx = self.bns(xx)
        xx = self.relu(xx)

        return xx

class DASI2(nn.Module):
    def __init__(self, in_features, in_features_high, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)

         self.skips_2 = conv_block(in_features=in_features_high,
                                   out_features=out_features,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   norm_type=None,
                                   activation=False)

         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        if x[1] != None:
            x[1] = self.skips_2(x[1])
            x[1] = F.interpolate(x[1], size=[x[0].size(2), x[0].size(3)], mode='bicubic', align_corners=True)
            x_high = torch.chunk(x[1], 4, dim=1)
        x_skip = self.skips(x[0])
        x[0] = self.skips(x[0])
        x_mid = torch.chunk(x[0], 4, dim=1)

        x0 = self.conv(torch.cat((x_mid[0], x_high[0]), dim=1))
        x1 = self.conv(torch.cat((x_mid[1], x_high[1]), dim=1))
        x2 = self.conv(torch.cat((x_mid[2], x_high[2]), dim=1))
        x3 = self.conv(torch.cat((x_mid[3], x_high[3]), dim=1))

        xx = torch.cat((x0, x1, x2, x3), dim=1)
        xx = self.tail_conv(xx)
        xx += x_skip
        xx = self.bns(xx)
        xx = self.relu(xx)

        return xx

class DASI2h(nn.Module):
    def __init__(self, in_features, in_features_high, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)

         self.skips_2 = conv_block(in_features=in_features_high,
                                   out_features=out_features,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   norm_type=None,
                                   activation=False)

         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        if x[1] != None:
            x[1] = self.skips_2(x[1])
            x[1] = F.interpolate(x[1], size=[x[0].size(2), x[0].size(3)], mode='bicubic', align_corners=True)
            x_high = torch.chunk(x[1], 4, dim=1)
        x_skip = self.skips(x[0])
        x[0] = self.skips(x[0])
        x_mid = torch.chunk(x[0], 4, dim=1)

        x0 = self.conv(torch.cat((x_mid[0], x_high[0]), dim=1))
        x1 = self.conv(torch.cat((x_mid[1], x_high[1]), dim=1))
        x2 = self.conv(torch.cat((x_mid[2], x_high[2]), dim=1))
        x3 = self.conv(torch.cat((x_mid[3], x_high[3]), dim=1))

        xx = torch.cat((x0, x1, x2, x3), dim=1)
        xx = self.tail_conv(xx)
        xx += x_skip
        xx = self.bns(xx)
        xx = self.relu(xx)

        return xx

class DASI1(nn.Module):
    def __init__(self, in_features_low, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)

         self.skips_3 = nn.Conv2d(in_features_low, out_features,
                                  kernel_size=3, stride=2, dilation=2, padding=2)
         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        if x[0] != None:
            x[0] = self.skips_3(x[0])
            x_low = torch.chunk(x[0], 4, dim=1)
        x_skip = self.skips(x[1])
        x[1] = self.skips(x[1])
        x_mid = torch.chunk(x[1], 4, dim=1)

        x0 = self.conv(torch.cat((x_mid[0], x_low[0]), dim=1))
        x1 = self.conv(torch.cat((x_mid[1], x_low[1]), dim=1))
        x2 = self.conv(torch.cat((x_mid[2], x_low[2]), dim=1))
        x3 = self.conv(torch.cat((x_mid[3], x_low[3]), dim=1))

        xx = torch.cat((x0, x1, x2, x3), dim=1)
        xx = self.tail_conv(xx)
        xx += x_skip
        xx = self.bns(xx)
        xx = self.relu(xx)

        return xx

class DASI1h(nn.Module):
    def __init__(self, in_features_low, in_features, out_features) -> None:
        super().__init__()
        self.bag = Bag()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=out_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.conv = nn.Sequential(
            conv_block(in_features=out_features // 2,
                       out_features=out_features // 4,
                       kernel_size=(1, 1),
                       padding=(0, 0),
                       norm_type=None,
                       activation=False)
        )
        self.bns = nn.BatchNorm2d(out_features)

        self.skips = conv_block(in_features=in_features,
                                out_features=out_features,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type=None,
                                activation=False)

        self.skips_3 = nn.Conv2d(in_features_low, out_features,
                                 kernel_size=3, stride=2, dilation=2, padding=2)
        # self.skips_3 = nn.Conv2d(in_features//2, out_features,
        #                          kernel_size=3, stride=2, dilation=1, padding=1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x):
        if x[0] != None:
            x[0] = self.skips_3(x[0])
            x_low = torch.chunk(x[0], 4, dim=1)
        x_skip = self.skips(x[1])
        x[1] = self.skips(x[1])
        x_mid = torch.chunk(x[1], 4, dim=1)

        x0 = self.conv(torch.cat((x_mid[0], x_low[0]), dim=1))
        x1 = self.conv(torch.cat((x_mid[1], x_low[1]), dim=1))
        x2 = self.conv(torch.cat((x_mid[2], x_low[2]), dim=1))
        x3 = self.conv(torch.cat((x_mid[3], x_low[3]), dim=1))

        xx = torch.cat((x0, x1, x2, x3), dim=1)
        xx = self.tail_conv(xx)
        xx += x_skip
        xx = self.bns(xx)
        xx = self.relu(xx)

        return xx

if __name__ == "__main__":
    x = torch.randn(1,3,160,160)
    model = FPIC(3,64)
    output = model(x)


    model1 = PIC(64,128)
    out1 = model1(output)
    model2 = PIC(128, 256)
    out2 = model2(out1)

    out = [out1,out2]
    model2 = Feature_selection(256)
    out3 = model2(out)
    # print(output.shape)
    # model = PatchMerging(128)
    print(output.shape)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
