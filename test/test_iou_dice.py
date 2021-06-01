import torch
import cv2
from utils.dice_loss import iou_pytorch
import numpy as np


def test_iou_pytorch():
    """
    Iouchch_pytorch

    Args:
    """
    y_pred = torch.zeros((4, 1, 256, 256))
    y_true = torch.zeros((4, 1, 256, 256))
    y_pred[:, :, :128, :2 * 256 // 3] = 1
    y_true[:, :, :128, 256 // 3:] = 1
    print(iou_pytorch(y_pred, y_true))