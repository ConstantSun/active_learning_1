"""
BlurPool layer inspired by
 - Kornia's Max_BlurPool2d
 - Making Convolutional Networks Shift-Invariant Again :cite:`zhang2019shiftinvar`

FIXME merge this impl with those in `anti_aliasing.py`

Hacked together by Chris Ha and Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from .padding import get_padding


class BlurPool2d(nn.Module):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    filt: Dict[str, torch.Tensor]

    def __init__(self, channels, filt_size=3, stride=2) -> None:
        """
        Initialize all channels.

        Args:
            self: (todo): write your description
            channels: (list): write your description
            filt_size: (int): write your description
            stride: (int): write your description
        """
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        pad_size = [get_padding(filt_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)
        self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs)  # for torchscript compat
        self.filt = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like: torch.Tensor):
        """
        Create a filter object from a tensor.

        Args:
            self: (str): write your description
            like: (str): write your description
            torch: (str): write your description
            Tensor: (str): write your description
        """
        blur_filter = (self._coeffs[:, None] * self._coeffs[None, :]).to(dtype=like.dtype, device=like.device)
        return blur_filter[None, None, :, :].repeat(self.channels, 1, 1, 1)

    def _apply(self, fn):
        """
        Apply fn fn fn fn fn fn fn fn.

        Args:
            self: (todo): write your description
            fn: (array): write your description
        """
        # override nn.Module _apply, reset filter cache if used
        self.filt = {}
        super(BlurPool2d, self)._apply(fn)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input_tensor: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
        """
        C = input_tensor.shape[1]
        blur_filt = self.filt.get(str(input_tensor.device), self._create_filter(input_tensor))
        return F.conv2d(
            self.padding(input_tensor), blur_filt, stride=self.stride, groups=C)
