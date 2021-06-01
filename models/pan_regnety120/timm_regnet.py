from models.pan_regnety120.timm.models.regnet import RegNet
import torch
import torch.nn as nn


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        """
        Create a stage from stage_list.

        Args:
            self: (todo): write your description
            stage_list: (list): write your description
            dilation_list: (list): write your description
        """
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


class utils:
    def __init__(self):
        """
        Initialize the object

        Args:
            self: (todo): write your description
        """
        pass

    def patch_first_conv(model, in_channels):
        """Change first convolution layer input channels.
        In case:
            in_channels == 1 or in_channels == 2 -> reuse original weights
            in_channels > 3 -> make random kaiming normal initialization
        """

        # get first conv
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                break

        # change input channels for first conv
        module.in_channels = in_channels
        weight = module.weight.detach()
        reset = False

        if in_channels == 1:
            weight = weight.sum(1, keepdim=True)
        elif in_channels == 2:
            weight = weight[:, :2] * (3.0 / 2.0)
        else:
            reset = True
            weight = torch.Tensor(
                module.out_channels,
                module.in_channels // module.groups,
                *module.kernel_size
            )

        module.weight = nn.parameter.Parameter(weight)
        if reset:
            module.reset_parameters()

    def replace_strides_with_dilation(module, dilation_rate):
        """Patch Conv2d modules replacing strides with dilation"""
        for mod in module.modules():
            if isinstance(mod, nn.Conv2d):
                mod.stride = (1, 1)
                mod.dilation = (dilation_rate, dilation_rate)
                kh, kw = mod.kernel_size
                mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

                # Kostyl for EfficientNet
                if hasattr(mod, "static_padding"):
                    mod.static_padding = nn.Identity()


class RegNetEncoder(RegNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        """
        Initialize the channel.

        Args:
            self: (todo): write your description
            out_channels: (int): write your description
            depth: (float): write your description
        """
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.head

    def get_stages(self):
        """
        Return a list of indra statements.

        Args:
            self: (todo): write your description
        """
        return [
            nn.Identity(),
            self.stem,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
        ]

    def forward(self, x):
        """
        Forward computation of the graph.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        """
        Reloads the state of a dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        state_dict.pop("head.fc.weight")
        state_dict.pop("head.fc.bias")
        super().load_state_dict(state_dict, **kwargs)


regnet_weights = {
    'timm-regnetx_002': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_002-e7e85e5c.pth',
    },
    'timm-regnetx_004': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_004-7d0e9424.pth',
    },
    'timm-regnetx_006':  {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_006-85ec1baa.pth',
    },
    'timm-regnetx_008': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_008-d8b470eb.pth',
    },
    'timm-regnetx_016': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_016-65ca972a.pth',
    },
    'timm-regnetx_032': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_032-ed0c7f7e.pth',
    },
    'timm-regnetx_040': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_040-73c2a654.pth',
    },
    'timm-regnetx_064': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_064-29278baa.pth',
    },
    'timm-regnetx_080': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_080-7c7fcab1.pth',
    },
    'timm-regnetx_120': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_120-65d5521e.pth',
    },
    'timm-regnetx_160': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_160-c98c4112.pth',
    },
    'timm-regnetx_320': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnetx_320-8ea38b93.pth',
    },
    'timm-regnety_002': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_002-e68ca334.pth',
    },
    'timm-regnety_004': {
         'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_004-0db870e6.pth',
    },
    'timm-regnety_006': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_006-c67e57ec.pth',
    },
    'timm-regnety_008': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_008-dc900dbe.pth',
    },
    'timm-regnety_016': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_016-54367f74.pth',
    },
    'timm-regnety_032': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/regnety_032_ra-7f2439f9.pth'
    },
    'timm-regnety_040': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_040-f0d569f9.pth'
    },
    'timm-regnety_064': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_064-0a48325c.pth'
    },
    'timm-regnety_080': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_080-e7f3eb93.pth',
    },
    'timm-regnety_120': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_120-721ba79a.pth',
    },
    'timm-regnety_160': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_160-d64013cd.pth',
    },
    'timm-regnety_320': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-regnet/regnety_320-ba464b29.pth'
    }
}

pretrained_settings = {}
for model_name, sources in regnet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }

# at this point I am too lazy to copy configs, so I just used the same configs from timm's repo


def _mcfg(**kwargs):
    """
    Return a multi - style configurations.

    Args:
    """
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


timm_regnet_encoders = {
    'timm-regnetx_002': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_002"],
        'params': {
            'out_channels': (3, 32, 24, 56, 152, 368),
            'cfg': _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13)
        },
    },
    'timm-regnetx_004': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_004"],
        'params': {
            'out_channels': (3, 32, 32, 64, 160, 384),
            'cfg': _mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22)
        },
    },
    'timm-regnetx_006': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_006"],
        'params': {
            'out_channels': (3, 32, 48, 96, 240, 528),
            'cfg': _mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16)
        },
    },
    'timm-regnetx_008': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_008"],
        'params': {
            'out_channels': (3, 32, 64, 128, 288, 672),
            'cfg': _mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16)
        },
    },
    'timm-regnetx_016': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_016"],
        'params': {
            'out_channels': (3, 32, 72, 168, 408, 912),
            'cfg': _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18)
        },
    },
    'timm-regnetx_032': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_032"],
        'params': {
            'out_channels': (3, 32, 96, 192, 432, 1008),
            'cfg': _mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25)
        },
    },
    'timm-regnetx_040': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_040"],
        'params': {
            'out_channels': (3, 32, 80, 240, 560, 1360),
            'cfg': _mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23)
        },
    },
    'timm-regnetx_064': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_064"],
        'params': {
            'out_channels': (3, 32, 168, 392, 784, 1624),
            'cfg': _mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17)
        },
    },
    'timm-regnetx_080': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_080"],
        'params': {
            'out_channels': (3, 32, 80, 240, 720, 1920),
            'cfg': _mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23)
        },
    },
    'timm-regnetx_120': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_120"],
        'params': {
            'out_channels': (3, 32, 224, 448, 896, 2240),
            'cfg': _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19)
        },
    },
    'timm-regnetx_160': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_160"],
        'params': {
            'out_channels': (3, 32, 256, 512, 896, 2048),
            'cfg': _mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22)
        },
    },
    'timm-regnetx_320': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnetx_320"],
        'params': {
            'out_channels': (3, 32, 336, 672, 1344, 2520),
            'cfg': _mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23)
        },
    },
    #regnety
    'timm-regnety_002': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_002"],
        'params': {
            'out_channels': (3, 32, 24, 56, 152, 368),
            'cfg': _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25)
        },
    },
    'timm-regnety_004': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_004"],
        'params': {
            'out_channels': (3, 32, 48, 104, 208, 440),
            'cfg': _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25)
        },
    },
    'timm-regnety_006': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_006"],
        'params': {
            'out_channels': (3, 32, 48, 112, 256, 608),
            'cfg': _mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25)
        },
    },
    'timm-regnety_008': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_008"],
        'params': {
            'out_channels': (3, 32, 64, 128, 320, 768),
            'cfg': _mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25)
        },
    },
    'timm-regnety_016': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_016"],
        'params': {
            'out_channels': (3, 32, 48, 120, 336, 888),
            'cfg': _mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25)
        },
    },
    'timm-regnety_032': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_032"],
        'params': {
            'out_channels': (3, 32, 72, 216, 576, 1512),
            'cfg': _mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25)
        },
    },
    'timm-regnety_040': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_040"],
        'params': {
            'out_channels': (3, 32, 128, 192, 512, 1088),
            'cfg': _mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25)
        },
    },
    'timm-regnety_064': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_064"],
        'params': {
            'out_channels': (3, 32, 144, 288, 576, 1296),
            'cfg': _mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25)
        },
    },
    'timm-regnety_080': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_080"],
        'params': {
            'out_channels': (3, 32, 168, 448, 896, 2016),
            'cfg': _mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25)
        },
    },
    'timm-regnety_120': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_120"],
        'params': {
            'out_channels': (3, 32, 224, 448, 896, 2240),
            'cfg': _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25)
        },
    },
    'timm-regnety_160': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_160"],
        'params': {
            'out_channels': (3, 32, 224, 448, 1232, 3024),
            'cfg': _mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25)
        },
    },
    'timm-regnety_320': {
        'encoder': RegNetEncoder,
        "pretrained_settings": pretrained_settings["timm-regnety_320"],
        'params': {
            'out_channels': (3, 32, 232, 696, 1392, 3712),
            'cfg': _mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25)
        },
    },
}
