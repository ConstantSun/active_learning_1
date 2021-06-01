import functools
import torch.utils.model_zoo as model_zoo
import numpy as np

from models.pan_regnety120.timm_regnet import timm_regnet_encoders

encoders = {}
encoders.update(timm_regnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):
    """
    Get encoder.

    Args:
        name: (str): write your description
        in_channels: (str): write your description
        depth: (int): write your description
        weights: (todo): write your description
    """
    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Avaliable options are: {}".format(
                weights, name, list(encoders[name]["pretrained_settings"].keys()),
            ))
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))
    encoder.set_in_channels(in_channels)
    return encoder


def get_encoder_names():
    """
    Returns a list of encoder names.

    Args:
    """
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    """
    Get preprocessing params.

    Args:
        encoder_name: (str): write your description
        pretrained: (bool): write your description
    """
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Avaliable pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    """
    Create a preprocessing function.

    Args:
        encoder_name: (str): write your description
        pretrained: (bool): write your description
    """
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)

def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):
    """
    Preprocess input.

    Args:
        x: (todo): write your description
        mean: (todo): write your description
        std: (todo): write your description
        input_space: (todo): write your description
        input_range: (todo): write your description
    """

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x
