""" Conv2d w/ Same Padding

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .padding import pad_same, get_padding_value


def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    """
    Conv2d conv2d convolution.

    Args:
        x: (todo): write your description
        weight: (str): write your description
        torch: (todo): write your description
        Tensor: (todo): write your description
        bias: (todo): write your description
        Optional: (todo): write your description
        torch: (todo): write your description
        Tensor: (todo): write your description
        stride: (int): write your description
        padding: (str): write your description
        dilation: (str): write your description
        groups: (array): write your description
    """
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        """
        Initialize the channel.

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
        """
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        """
        Bias.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    """
    Create a convolution padding.

    Args:
        in_chs: (int): write your description
        out_chs: (todo): write your description
        kernel_size: (int): write your description
    """
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


