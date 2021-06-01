""" Conv2d + BN + Act

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import convert_norm_act_type


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, act_layer=nn.ReLU, apply_act=True,
                 drop_block=None, aa_layer=None):
        """
        Initialize the layer.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            padding: (str): write your description
            dilation: (todo): write your description
            groups: (list): write your description
            norm_layer: (int): write your description
            nn: (int): write your description
            BatchNorm2d: (todo): write your description
            norm_kwargs: (dict): write your description
            act_layer: (todo): write your description
            nn: (int): write your description
            ReLU: (todo): write your description
            apply_act: (todo): write your description
            drop_block: (todo): write your description
            aa_layer: (todo): write your description
        """
        super(ConvBnAct, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None

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
        Return the channel channels.

        Args:
            self: (todo): write your description
        """
        return self.conv.out_channels

    def forward(self, x):
        """
        Forward forward. forward

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x
