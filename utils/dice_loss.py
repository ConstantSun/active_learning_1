import torch
from torch.autograd import Function
import numpy as np
import logging


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        """
        Perform computation

        Args:
            self: (todo): write your description
            input: (array): write your description
            target: (array): write your description
        """
        self.save_for_backward(input, target)
        eps = 0.0001
        # print("input dice shape: {}".format(input.shape))
        # print("target dice shape: {}".format(target.shape))
        # logging.info("input dice shape: {}".format(input.shape))
        # logging.info("target dice shape: {}".format(target.shape))

        self.inter = torch.dot(input.reshape(-1), target.reshape(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """
        Perform backward backward pass.

        Args:
            self: (todo): write your description
            grad_output: (bool): write your description
        """

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target, _device):
    """Dice coeff for batches"""
    if input.is_cuda:
        # cuda_ = torch.device(_device)
        s = torch.FloatTensor(1).to(device = _device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        # print(f"c[0], c[1] shape: {c[0].shape}, {c[1].shape}")
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

SMOOTH = 1e-6


def iou_pytorch(input: torch.Tensor, target: torch.Tensor):
    """
    Computes the tensor.

    Args:
        input: (array): write your description
        torch: (todo): write your description
        Tensor: (todo): write your description
        target: (array): write your description
        torch: (todo): write your description
        Tensor: (todo): write your description
    """
    eps = 0.0001
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    union = torch.sum(input) + torch.sum(target) + eps - inter

    t = (inter.float() + eps) / union.float()
    return t


def iou_numpy(outputs: np.array, labels: np.array):
    """
    Iou_numpy arrays

    Args:
        outputs: (array): write your description
        np: (todo): write your description
        array: (array): write your description
        labels: (list): write your description
        np: (todo): write your description
        array: (array): write your description
    """
    outputs = outputs.squeeze(1)
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()