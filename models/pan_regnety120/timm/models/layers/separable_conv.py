""" Depthwise Separable Conv Modules

Basic DWS convs. Other variations of DWS exist with batch norm or activations between the
DW and PW convs such as the Depthwise modules in MobileNetV2 / EfficientNet and Xception.

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .create_conv2d import create_conv2d
from .create_norm_act import convert_norm_act_type


class SeparableConvBnAct(nn.Module):
    """ Separable Conv w/ trailing Norm and Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 act_layer=nn.ReLU, apply_act=True, drop_block=None):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            dilation: (todo): write your description
            padding: (str): write your description
            bias: (float): write your description
            channel_multiplier: (todo): write your description
            pw_kernel_size: (int): write your description
            norm_layer: (int): write your description
            nn: (int): write your description
            BatchNorm2d: (todo): write your description
            norm_kwargs: (dict): write your description
            act_layer: (todo): write your description
            nn: (int): write your description
            ReLU: (todo): write your description
            apply_act: (todo): write your description
            drop_block: (todo): write your description
        """
        super(SeparableConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        norm_act_layer, norm_act_args = convert_norm_act_type(norm_layer, act_layer, norm_kwargs)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block, **norm_act_args)

    @property
    def in_channels(self):
        """
        List of : list of channels in this channel.

        Args:
            self: (todo): write your description
        """
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        """
        : return : a : class channels.

        Args:
            self: (todo): write your description
        """
        return self.conv_pw.out_channels

    def forward(self, x):
        """
        Forward computation. forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            dilation: (todo): write your description
            padding: (str): write your description
            bias: (float): write your description
            channel_multiplier: (todo): write your description
            pw_kernel_size: (int): write your description
        """
        super(SeparableConv2d, self).__init__()

        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

    @property
    def in_channels(self):
        """
        List of : list of channels in this channel.

        Args:
            self: (todo): write your description
        """
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        """
        : return : a : class channels.

        Args:
            self: (todo): write your description
        """
        return self.conv_pw.out_channels

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x