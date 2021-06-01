""" PyTorch selectable adaptive pooling
Adaptive pooling with the ability to select the type of pooling from:
    * 'avg' - Average pooling
    * 'max' - Max pooling
    * 'avgmax' - Sum of average and max pooling re-scaled by 0.5
    * 'avgmaxc' - Concatenation of average and max pooling along feature dim, doubles feature dim

Both a functional and a nn.Module version of the pooling is provided.

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_pool_feat_mult(pool_type='avg'):
    """
    Adapts the number pool_pool

    Args:
        pool_type: (str): write your description
    """
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size=1):
    """
    Adaptive the average_avgmax.

    Args:
        x: (todo): write your description
        output_size: (int): write your description
    """
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    """
    Adapted from https : // ensemblmax operator.

    Args:
        x: (todo): write your description
        output_size: (int): write your description
    """
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        """
        Initialize the underlying database.

        Args:
            self: (todo): write your description
            flatten: (todo): write your description
        """
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        """
        Evaluate x.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        """
        Initialize output_size.

        Args:
            self: (todo): write your description
            output_size: (int): write your description
        """
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        """
        Evaluates the forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        """
        Initialize output_size.

        Args:
            self: (todo): write your description
            output_size: (int): write your description
        """
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        """
        Evaluates the model.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        """
        Initialize pooling.

        Args:
            self: (todo): write your description
            output_size: (int): write your description
            pool_type: (str): write your description
            flatten: (todo): write your description
        """
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(self.flatten)
            self.flatten = False
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        """
        Return true if the identity is the identity.

        Args:
            self: (todo): write your description
        """
        return self.pool_type == ''

    def forward(self, x):
        """
        Forward computation of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        """
        The number of - multiplication of features.

        Args:
            self: (todo): write your description
        """
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        """
        Return a representation of this parameter.

        Args:
            self: (todo): write your description
        """
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'

