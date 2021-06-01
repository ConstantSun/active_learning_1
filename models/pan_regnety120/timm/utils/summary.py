""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict


def get_outdir(path, *paths, inc=False):
    """
    Retrieve output directory of path.

    Args:
        path: (str): write your description
        paths: (str): write your description
        inc: (int): write your description
    """
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    """
    Update summary of training summary.

    Args:
        epoch: (int): write your description
        train_metrics: (dict): write your description
        eval_metrics: (dict): write your description
        filename: (str): write your description
        write_header: (bool): write your description
    """
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
