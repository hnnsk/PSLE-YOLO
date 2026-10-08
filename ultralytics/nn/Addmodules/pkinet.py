import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard conv-bn-act block reused by lightweight custom modules."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CAA(nn.Module):
    """
    Context Anchor Attention from PKINet (CVPR 2024).

    The original attention block produces an attention factor. In this Ultralytics
    branch it is exposed as a standalone layer, so forward() returns x * attn.
    """

    def __init__(self, c1, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        if h_kernel_size % 2 == 0 or v_kernel_size % 2 == 0:
            raise ValueError("CAA kernel sizes must be odd to preserve spatial size.")

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1 = Conv(c1, c1, k=1, s=1, p=0)
        self.h_conv = Conv(c1, c1, k=(1, h_kernel_size), s=1, p=(0, h_kernel_size // 2), g=c1)
        self.v_conv = Conv(c1, c1, k=(v_kernel_size, 1), s=1, p=(v_kernel_size // 2, 0), g=c1)
        self.conv2 = Conv(c1, c1, k=1, s=1, p=0, act=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.conv1(attn)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn
