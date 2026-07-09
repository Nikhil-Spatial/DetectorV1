from configs import LAMBDA_COORD, LAMBDA_NOOBJ, S, B
from uilts.py import IoU
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pass

class WidthHeightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pass

class ObjectConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pass
    
class NoObjectConfidenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pass

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pass

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _find_responsibility_(self, pred, target, row, col):
        """Returns 0 if the first bounding box predictor is responsible for
        a prediction, and 1 if the second is responsible."""
        target_bbox = target[20:24]
        pred_bbox_1 = target[20:24]
        pred_bbox_2 = target[25:29]

        IoU_1 = IoU()
        IoU_2 = IoU(pred[25:29])

    def forward(self, pred, target):
        for i in list(range(7)):
            for j in list(range(7)):
