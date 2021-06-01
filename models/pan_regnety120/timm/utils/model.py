""" Model / state_dict utils

Hacked together by / Copyright 2020 Ross Wightman
"""
from .model_ema import ModelEma


def unwrap_model(model):
    """
    Unwrap model. model. model.

    Args:
        model: (todo): write your description
    """
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    """
    Get the state dictionary.

    Args:
        model: (str): write your description
        unwrap_fn: (str): write your description
        unwrap_model: (str): write your description
    """
    return unwrap_fn(model).state_dict()
