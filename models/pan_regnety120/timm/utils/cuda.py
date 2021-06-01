""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        """
        Perform an optimizer.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            optimizer: (todo): write your description
            clip_grad: (bool): write your description
            parameters: (todo): write your description
            create_graph: (bool): write your description
        """
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), clip_grad)
        optimizer.step()

    def state_dict(self):
        """
        : return : class : ~ /.

        Args:
            self: (todo): write your description
        """
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        """
        Initialize the game.

        Args:
            self: (todo): write your description
        """
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        """
        Perform an optimizer.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            optimizer: (todo): write your description
            clip_grad: (bool): write your description
            parameters: (todo): write your description
            create_graph: (bool): write your description
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        """
        : return : class : scaler. scaler. state.

        Args:
            self: (todo): write your description
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary.

        Args:
            self: (todo): write your description
            state_dict: (dict): write your description
        """
        self._scaler.load_state_dict(state_dict)
