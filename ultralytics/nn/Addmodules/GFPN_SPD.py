import os.path
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .lsk import LSKFEM, LSKblock
from .pkinet import CAA


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

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

class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish'
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k, 1, 1)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='silu',
                 shortcut=True
                 ):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, 1, 1)
        self.conv2 = RepConv(ch_in, ch_hidden, kernel_size=3, stride=1, act=act)
        self.shortcut = shortcut


    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 act='swish',
                 spp=False
                 ):
        super(CSPStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1, 1, 0)
        self.conv2 = Conv(ch_in, ch_mid, 1, 1, 0)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True,
                                           ))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1, 1, 0)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)

        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.SiLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super().__init__()
        self.dw = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_planes,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True)
        self.pw = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.act = nn.SiLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x

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


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super().__init__()
        self.groups = factor
        assert channels % self.groups == 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.5, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes//4
        inter_planes = max(1, in_planes // map_reduce)
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            DepthwiseSeparableConv(2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=1),
            # BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            # BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            DepthwiseSeparableConv(
                2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=2, dilation=2, relu=False
            )
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=1),
            # BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            # BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            DepthwiseSeparableConv(
                2 * inter_planes, out_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False
            )
        )

        # self.branch3 = SCAM(in_planes,out_planes)

        # self.ConvLinear = BasicConv(6 * inter_planes, out_planes//4, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.SiLU(inplace=True)
        self.conv = DepthwiseSeparableConv(out_planes, out_planes, kernel_size=3, stride=2, padding=1, relu=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)

        # out = torch.cat((x0, x1, x2), 1)
        # out = self.ConvLinear(out)
        # short = self.shortcut(x)
        # out = out * self.scale + short
        out = (x0 + x1 + x2)*self.scale +self.shortcut(x)
        out = self.relu(out)
        out = self.conv(out)

        return out


class DFEM(FEM):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.5, map_reduce=16):
        super().__init__(in_planes, out_planes, stride=stride, scale=scale, map_reduce=map_reduce)


class GFPNNeck(nn.Module):
    # default_act = nn.SiLU()  # default activation
    def __init__(
        self,
        depth=1.0,
        hidden_ratio=1.0,
        in_channels=[256, 512, 256],
        out_channels=[64, 128, 256],
        act='silu',
        spp=False,
        block_name='BasicBlock_3x3_Reverse',
        dfem_type='lsk',
        dfem_reduce=16,
        dfem_high_res_only=False,
        use_ema=True,
        downsample_mode='spd',
        attn_type='ema',
    ):
        super(GFPNNeck, self).__init__()
        dfem_type = str(dfem_type).lower()
        if dfem_type == 'dfem':
            def make_dfem(c1, c2):
                return DFEM(c1, c2, map_reduce=dfem_reduce)
        elif dfem_type == 'lsk':
            def make_dfem(c1, c2):
                return LSKFEM(c1, c2)
        elif dfem_type in {'none', 'conv'}:
            def make_dfem(c1, c2):
                # Fair ablations replace DFEM with a plain stride-2 conv reducer.
                return Conv(c1, c2, 3, 2, 1)
        else:
            raise ValueError(f"Unsupported dfem_type: {dfem_type}")

        if dfem_high_res_only:
            self.cv_40_1 = Conv(in_channels[1], in_channels[1], 3, 2, 1)
            self.cv_40_2 = Conv(out_channels[1], out_channels[1], 3, 2, 1)
            self.cv_80 = make_dfem(in_channels[0], in_channels[0])
        else:
            self.cv_40_1 = make_dfem(in_channels[1], in_channels[1])
            self.cv_40_2 = make_dfem(out_channels[1], out_channels[1])
            self.cv_80 = make_dfem(in_channels[0], in_channels[0])

        self.cv_160_un = Conv(out_channels[0], out_channels[0], 3, 1, 1)
        self.cv_80_un = Conv(out_channels[1], out_channels[1], 3, 1, 1)

        downsample_mode = str(downsample_mode).lower()
        if downsample_mode in {'maxpool', 'mp'}:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ds_mult = 1
        else:
            self.downsample = SPD(2)
            ds_mult = 4
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.FB_40_1 = CSPStage(block_name, in_channels[1] + in_channels[2], hidden_ratio, out_channels[2], round(3 * depth), act='silu', spp=spp)
        self.FB_40_1 = C2f(in_channels[1] + in_channels[2], out_channels[2])
        # self.FB_80_1 = CSPStage(block_name, in_channels[1] + in_channels[0]+in_channels[2], hidden_ratio, out_channels[1], round(3*depth), act='silu', spp=spp)
        self.FB_80_1 = C2f(in_channels[1] + in_channels[0]+in_channels[2], out_channels[1])
        # self.FB_160_1 = CSPStage(block_name, in_channels[0] + 128, hidden_ratio, out_channels[0], round(3*depth), act='silu', spp=spp)
        self.FB_160_1 = C2f(in_channels[0] + 128, out_channels[0])
        # self.FB_80_2 = CSPStage(block_name, 4*out_channels[0] + out_channels[1], hidden_ratio, out_channels[1],round(3 * depth), act='silu', spp=spp)
        self.FB_80_2 = C2f(ds_mult * out_channels[0] + out_channels[1], out_channels[1])
        # self.FB_40_2 = CSPStage(block_name, out_channels[1] + in_channels[1]+out_channels[2], hidden_ratio, out_channels[2], round(3 * depth),act='silu', spp=spp)
        self.FB_40_2 = C2f(out_channels[2] + out_channels[1] + ds_mult * out_channels[1], out_channels[2])

        self.lsk1 = LSKblock(out_channels[1])
        self.lsk2 = LSKblock(out_channels[2])

        attn_type = str(attn_type).lower()
        if not use_ema or attn_type in {'none', 'identity'}:
            make_attention = lambda c: nn.Identity()
        elif attn_type == 'ema':
            make_attention = lambda c: EMA(c)
        elif attn_type == 'caa':
            make_attention = lambda c: CAA(c)
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        self.head0 = make_attention(out_channels[0])
        self.head1 = make_attention(out_channels[1])
        self.head2 = make_attention(out_channels[2])

    def _restore_downsample_for_legacy_checkpoint(self):
        """Backfill missing downsample for checkpoints deserialized from older GFPNNeck definitions."""
        if hasattr(self, 'downsample'):
            return

        # Infer the historical downsample mode from the expected input width of FB_80_2.
        fb80_in = getattr(getattr(self.FB_80_2, 'cv1', None), 'conv', None)
        fb80_in = getattr(fb80_in, 'in_channels', None)
        cv160_out = getattr(getattr(self.cv_160_un, 'conv', None), 'out_channels', None)

        if fb80_in is not None and cv160_out:
            residual_channels = fb80_in - cv160_out
            ds_mult = residual_channels // cv160_out if residual_channels >= cv160_out else 1
        else:
            ds_mult = 4

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) if ds_mult == 1 else SPD(2)

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        self._restore_downsample_for_legacy_checkpoint()

        #  backbone  x2=160*160 x1=80*80 x0 = 40*40
        [x2, x1, x0] = out_features

        # node 40*40
        x13 = self.cv_40_1(x1)
        x3 = torch.cat([x0, x13], 1)
        x3 = self.FB_40_1(x3)

        # node x4
        x34 = self.upsample(x3)
        x24 = self.cv_80(x2)
        x4 = torch.cat([x1, x24, x34], 1)
        x4 = self.FB_80_1(x4)

        # node x5
        x45 = self.upsample(x4)
        x5 = torch.cat([x2, x45], 1)
        x5 = self.FB_160_1(x5)
        x5_n = self.head0(x5)
        # node x8
        # x8 = x5

        # node x7
        x57 = self.cv_160_un(x5)
        x57 = self.downsample(x57)
        x7 = torch.cat([x4, x57], 1)
        x7 = self.FB_80_2(x7)
        x7 = self.lsk1(x7)
        x7_n = self.head1(x7)

        # node x6
        x46 = self.cv_40_2(x4)
        x76 = self.cv_80_un(x7)
        x76 = self.downsample(x76)
        x6 = torch.cat([x3, x46, x76], 1)
        x6 = self.FB_40_2(x6)
        x6 = self.lsk2(x6)
        x6 = self.head2(x6)

        # x5=160*160  x7=80*80  x6=40*40
        outputs = [x5_n, x7_n, x6]
        return outputs



# -----------------------------------------------------------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


if __name__ == "__main__":
    # pretrained_path = "/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/pvt_pth/pvt_v2_b2.pth"
    model = GFPNNeck(depth=1.0,hidden_ratio=1.0,in_channels=[256, 512, 256],out_channels=[64, 128, 256],act='silu',spp=False,block_name='BasicBlock_3x3_Reverse')
    x2 = torch.rand((1, 256, 160, 160))
    x1 = torch.rand((1, 512, 80, 80))
    x0 = torch.rand((1, 256, 40, 40))
    x = [x2,x1,x0]
    ys = model(x)
    print('ok')
