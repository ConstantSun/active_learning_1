""" Distributed training/validation utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import distributed as dist

from .model import unwrap_model


def reduce_tensor(tensor, n):
    """
    Reduce the tensor.

    Args:
        tensor: (todo): write your description
        n: (todo): write your description
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    """
    Distribute all the given model.

    Args:
        model: (todo): write your description
        world_size: (int): write your description
        reduce: (str): write your description
    """
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)
