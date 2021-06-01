""" ReXNet

A PyTorch impl of `ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network` -
https://arxiv.org/abs/2007.00992

Adapted from original impl at https://github.com/clovaai/rexnet
Copyright (c) 2020-present NAVER Corp. MIT license

Changes for timm, feature extraction, and rounded channel variant hacked together by Ross Wightman
Copyright 2020 Ross Wightman
"""

import torch.nn as nn
from math import ceil

from models.pan_regnety120.timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ClassifierHead, create_act_layer, ConvBnAct, DropPath
from .registry import register_model
from .efficientnet_builder import efficientnet_init_weights


def _cfg(url=''):
    """
    Gets configuration.

    Args:
        url: (str): write your description
    """
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
    }


default_cfgs = dict(
    rexnet_100=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_100-1b4dddf4.pth'),
    rexnet_130=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_130-590d768e.pth'),
    rexnet_150=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_150-bd1a6aa8.pth'),
    rexnet_200=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rexnet/rexnetv1_200-8c0b7f2d.pth'),
    rexnetr_100=_cfg(
        url=''),
    rexnetr_130=_cfg(
        url=''),
    rexnetr_150=_cfg(
        url=''),
    rexnetr_200=_cfg(
        url=''),
)


def make_divisible(v, divisor=8, min_value=None):
    """
    Make a divisible value.

    Args:
        v: (todo): write your description
        divisor: (todo): write your description
        min_value: (float): write your description
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v


class SEWithNorm(nn.Module):

    def __init__(self, channels, se_ratio=1 / 12., act_layer=nn.ReLU, divisor=1, reduction_channels=None,
                 gate_layer='sigmoid'):
        """
        Initialize the layer.

        Args:
            self: (todo): write your description
            channels: (list): write your description
            se_ratio: (todo): write your description
            act_layer: (todo): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            divisor: (todo): write your description
            reduction_channels: (todo): write your description
            gate_layer: (todo): write your description
        """
        super(SEWithNorm, self).__init__()
        reduction_channels = reduction_channels or make_divisible(int(channels * se_ratio), divisor=divisor)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(reduction_channels)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        """
        Evaluate forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class LinearBottleneck(nn.Module):
    def __init__(self, in_chs, out_chs, stride, exp_ratio=1.0, se_ratio=0., ch_div=1, drop_path=None):
        """
        Initialize the layer.

        Args:
            self: (todo): write your description
            in_chs: (int): write your description
            out_chs: (str): write your description
            stride: (int): write your description
            exp_ratio: (float): write your description
            se_ratio: (todo): write your description
            ch_div: (todo): write your description
            drop_path: (str): write your description
        """
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_chs <= out_chs
        self.in_channels = in_chs
        self.out_channels = out_chs

        if exp_ratio != 1.:
            dw_chs = make_divisible(round(in_chs * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvBnAct(in_chs, dw_chs, act_layer="swish")
        else:
            dw_chs = in_chs
            self.conv_exp = None

        self.conv_dw = ConvBnAct(dw_chs, dw_chs, 3, stride=stride, groups=dw_chs, apply_act=False)
        self.se = SEWithNorm(dw_chs, se_ratio=se_ratio, divisor=ch_div) if se_ratio > 0. else None
        self.act_dw = nn.ReLU6()

        self.conv_pwl = ConvBnAct(dw_chs, out_chs, 1, apply_act=False)
        self.drop_path = drop_path

    def feat_channels(self, exp=False):
        """
        The channel channels.

        Args:
            self: (todo): write your description
            exp: (int): write your description
        """
        return self.conv_dw.out_channels if exp else self.out_channels

    def forward(self, x):
        """
        Forward computation on x forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.use_shortcut:
            x[:, 0:self.in_channels] += shortcut
        return x


def _block_cfg(width_mult=1.0, depth_mult=1.0, initial_chs=16, final_chs=180, se_ratio=0., ch_div=1):
    """
    Calculate the block block block block.

    Args:
        width_mult: (int): write your description
        depth_mult: (int): write your description
        initial_chs: (todo): write your description
        final_chs: (bool): write your description
        se_ratio: (bool): write your description
        ch_div: (int): write your description
    """
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    layers = [ceil(element * depth_mult) for element in layers]
    strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    depth = sum(layers[:]) * 3
    base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs

    # The following channel configuration is a simple instance to make each layer become an expand layer.
    out_chs_list = []
    for i in range(depth // 3):
        out_chs_list.append(make_divisible(round(base_chs * width_mult), divisor=ch_div))
        base_chs += final_chs / (depth // 3 * 1.0)

    se_ratios = [0.] * (layers[0] + layers[1]) + [se_ratio] * sum(layers[2:])

    return list(zip(out_chs_list, exp_ratios, strides, se_ratios))


def _build_blocks(
        block_cfg, prev_chs, width_mult, ch_div=1, drop_path_rate=0., feature_location='bottleneck'):
    """
    Builds a list of features.

    Args:
        block_cfg: (todo): write your description
        prev_chs: (str): write your description
        width_mult: (int): write your description
        ch_div: (str): write your description
        drop_path_rate: (str): write your description
        feature_location: (bool): write your description
    """
    feat_exp = feature_location == 'expansion'
    feat_chs = [prev_chs]
    feature_info = []
    curr_stride = 2
    features = []
    num_blocks = len(block_cfg)
    for block_idx, (chs, exp_ratio, stride, se_ratio) in enumerate(block_cfg):
        if stride > 1:
            fname = 'stem' if block_idx == 0 else f'features.{block_idx - 1}'
            if block_idx > 0 and feat_exp:
                fname += '.act_dw'
            feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=fname)]
            curr_stride *= stride
        block_dpr = drop_path_rate * block_idx / (num_blocks - 1)  # stochastic depth linear decay rule
        drop_path = DropPath(block_dpr) if block_dpr > 0. else None
        features.append(LinearBottleneck(
            in_chs=prev_chs, out_chs=chs, exp_ratio=exp_ratio, stride=stride, se_ratio=se_ratio,
            ch_div=ch_div, drop_path=drop_path))
        prev_chs = chs
        feat_chs += [features[-1].feat_channels(feat_exp)]
    pen_chs = make_divisible(1280 * width_mult, divisor=ch_div)
    feature_info += [dict(
        num_chs=pen_chs if feat_exp else feat_chs[-1], reduction=curr_stride,
        module=f'features.{len(features) - int(not feat_exp)}')]
    features.append(ConvBnAct(prev_chs, pen_chs, act_layer="swish"))
    return features, feature_info


class ReXNetV1(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32,
                 initial_chs=16, final_chs=180, width_mult=1.0, depth_mult=1.0, se_ratio=1/12.,
                 ch_div=1, drop_rate=0.2, drop_path_rate=0., feature_location='bottleneck'):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_chans: (int): write your description
            num_classes: (int): write your description
            global_pool: (todo): write your description
            output_stride: (str): write your description
            initial_chs: (todo): write your description
            final_chs: (todo): write your description
            width_mult: (int): write your description
            depth_mult: (float): write your description
            se_ratio: (todo): write your description
            ch_div: (todo): write your description
            drop_rate: (float): write your description
            drop_path_rate: (str): write your description
            feature_location: (str): write your description
        """
        super(ReXNetV1, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        assert output_stride == 32  # FIXME support dilation
        stem_base_chs = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base_chs * width_mult), divisor=ch_div)
        self.stem = ConvBnAct(in_chans, stem_chs, 3, stride=2, act_layer='swish')

        block_cfg = _block_cfg(width_mult, depth_mult, initial_chs, final_chs, se_ratio, ch_div)
        features, self.feature_info = _build_blocks(
            block_cfg, stem_chs, width_mult, ch_div, drop_path_rate, feature_location)
        self.num_features = features[-1].out_channels
        self.features = nn.Sequential(*features)

        self.head = ClassifierHead(self.num_features, num_classes, global_pool, drop_rate)

        efficientnet_init_weights(self)

    def get_classifier(self):
        """
        Returns the classifier.

        Args:
            self: (str): write your description
        """
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        """
        Reset the classifier.

        Args:
            self: (todo): write your description
            num_classes: (int): write your description
            global_pool: (todo): write your description
        """
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        """
        Forward features.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.stem(x)
        x = self.features(x)
        return x

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_rexnet(variant, pretrained, **kwargs):
    """
    Vel factory function

    Args:
        variant: (todo): write your description
        pretrained: (bool): write your description
    """
    feature_cfg = dict(flatten_sequential=True)
    if kwargs.get('feature_location', '') == 'expansion':
        feature_cfg['feature_cls'] = 'hook'
    return build_model_with_cfg(
        ReXNetV1, variant, pretrained, default_cfg=default_cfgs[variant], feature_cfg=feature_cfg, **kwargs)


@register_model
def rexnet_100(pretrained=False, **kwargs):
    """ReXNet V1 1.0x"""
    return _create_rexnet('rexnet_100', pretrained, **kwargs)


@register_model
def rexnet_130(pretrained=False, **kwargs):
    """ReXNet V1 1.3x"""
    return _create_rexnet('rexnet_130', pretrained, width_mult=1.3, **kwargs)


@register_model
def rexnet_150(pretrained=False, **kwargs):
    """ReXNet V1 1.5x"""
    return _create_rexnet('rexnet_150', pretrained, width_mult=1.5, **kwargs)


@register_model
def rexnet_200(pretrained=False, **kwargs):
    """ReXNet V1 2.0x"""
    return _create_rexnet('rexnet_200', pretrained, width_mult=2.0, **kwargs)


@register_model
def rexnetr_100(pretrained=False, **kwargs):
    """ReXNet V1 1.0x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_100', pretrained, ch_div=8, **kwargs)


@register_model
def rexnetr_130(pretrained=False, **kwargs):
    """ReXNet V1 1.3x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_130', pretrained, width_mult=1.3, ch_div=8, **kwargs)


@register_model
def rexnetr_150(pretrained=False, **kwargs):
    """ReXNet V1 1.5x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_150', pretrained, width_mult=1.5, ch_div=8, **kwargs)


@register_model
def rexnetr_200(pretrained=False, **kwargs):
    """ReXNet V1 2.0x w/ rounded (mod 8) channels"""
    return _create_rexnet('rexnetr_200', pretrained, width_mult=2.0, ch_div=8, **kwargs)
