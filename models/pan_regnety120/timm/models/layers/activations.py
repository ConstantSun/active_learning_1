""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the data.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return swish(x, self.inplace)


def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(Mish, self).__init__()

    def forward(self, x):
        """
        Forward function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return mish(x)


def sigmoid(x, inplace: bool = False):
    """
    Sigmoid of x.

    Args:
        x: (todo): write your description
        inplace: (bool): write your description
    """
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the sigmoid.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace: bool = False):
    """
    Evaluate of x.

    Args:
        x: (todo): write your description
        inplace: (bool): write your description
    """
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the data.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool = False):
    """
    Hard_swish ( x.

    Args:
        x: (array): write your description
        inplace: (bool): write your description
    """
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    """
    Hard_sigmoid function ( x ). sigmoid of x.

    Args:
        x: (todo): write your description
        inplace: (bool): write your description
    """
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return hard_sigmoid(x, self.inplace)


def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return hard_mish(x, self.inplace)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        """
        Initialize the internal values.

        Args:
            self: (todo): write your description
            num_parameters: (int): write your description
            init: (str): write your description
            inplace: (todo): write your description
        """
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
        """
        return F.prelu(input, self.weight)


def gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Returns a tensor.

    Args:
        x: (todo): write your description
        torch: (todo): write your description
        Tensor: (todo): write your description
        inplace: (bool): write your description
    """
    return F.gelu(x)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        """
        Initialize the internal data.

        Args:
            self: (todo): write your description
            inplace: (todo): write your description
        """
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            torch: (todo): write your description
            Tensor: (todo): write your description
        """
        return F.gelu(input)
