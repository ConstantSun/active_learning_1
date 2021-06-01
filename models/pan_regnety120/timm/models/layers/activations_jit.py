""" Activations

A collection of jit-scripted activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

All jit scripted activations are lacking in-place variations on purpose, scripted kernel fusion does not
currently work across in-place op boundaries, thus performance is equal to or less than the non-scripted
versions if they contain in-place ops.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.jit.script
def swish_jit(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul(x.sigmoid())


@torch.jit.script
def mish_jit(x, _inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the __init__.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(SwishJit, self).__init__()

    def forward(self, x):
        """
        Forward computation of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return swish_jit(x)


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(MishJit, self).__init__()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool = False):
    """
    Sigmoid function that x.

    Args:
        x: (todo): write your description
        inplace: (bool): write your description
    """
    # return F.relu6(x + 3.) / 6.
    return (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSigmoidJit(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool = False):
    """
    Swish_swish.

    Args:
        x: (int): write your description
        inplace: (bool): write your description
    """
    # return x * (F.relu6(x + 3.) / 6)
    return x * (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return hard_swish_jit(x)


@torch.jit.script
def hard_mish_jit(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(HardMishJit, self).__init__()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return hard_mish_jit(x)
