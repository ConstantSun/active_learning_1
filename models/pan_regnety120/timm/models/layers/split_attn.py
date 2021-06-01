""" Split Attention Conv2d (for ResNeSt Models)

Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl at https://github.com/zhanghang1989/ResNeSt

Modified for torchscript compat, performance, and consistency with timm by Ross Wightman
"""
import torch
import torch.nn.functional as F
from torch import nn


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        """
        Initialize the cardinality.

        Args:
            self: (todo): write your description
            radix: (float): write your description
            cardinality: (array): write your description
        """
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttnConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, radix=2, reduction_factor=4,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            padding: (str): write your description
            dilation: (todo): write your description
            groups: (list): write your description
            bias: (float): write your description
            radix: (float): write your description
            reduction_factor: (todo): write your description
            act_layer: (todo): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            norm_layer: (int): write your description
            drop_block: (todo): write your description
        """
        super(SplitAttnConv2d, self).__init__()
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        attn_chs = max(in_channels * radix // reduction_factor, 32)

        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer is not None else None
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer is not None else None
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    @property
    def in_channels(self):
        """
        : return : class.

        Args:
            self: (todo): write your description
        """
        return self.conv.in_channels

    @property
    def out_channels(self):
        """
        The number of channels that are currently out_channels.

        Args:
            self: (todo): write your description
        """
        return self.fc1.out_channels

    def forward(self, x):
        """
        Compute forward computation.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        x = self.conv(x)
        if self.bn0 is not None:
            x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = F.adaptive_avg_pool2d(x_gap, 1)
        x_gap = self.fc1(x_gap)
        if self.bn1 is not None:
            x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()
