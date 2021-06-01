import torch.nn as nn
from .modules import Flatten, Activation
import baal.bayesian.dropout as baal


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, is_dropout = True):        
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            activation: (str): write your description
            upsampling: (todo): write your description
            is_dropout: (bool): write your description
        """
        dropout = baal.Dropout(p=0.5)
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        if is_dropout:
            super().__init__(conv2d, dropout, upsampling, activation)
        else:
            super().__init__(conv2d, upsampling, activation)
