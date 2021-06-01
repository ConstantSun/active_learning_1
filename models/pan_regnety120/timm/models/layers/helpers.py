""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
from torch._six import container_abcs


# From PyTorch internals
def _ntuple(n):
    """
    Convert a tuple.

    Args:
        n: (todo): write your description
    """
    def parse(x):
        """
        Convert a tuple or tuple.

        Args:
            x: (str): write your description
        """
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple





