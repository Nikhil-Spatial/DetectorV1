from configs import LAMBDA_COORD, LAMBDA_NOOBJ, S, C
import torch.nn as nn
from utils import IoU
import torch

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def _find_responsible_(self, pred_cell, target_cell, row, col):
        """Returns 0 if the first bounding box predictor is responsible for
        a prediction, and 1 if the second is responsible."""
        target_bbox = target_cell[C:C+4]
        pred_bbox_1 = pred_cell[C:C+4]
        pred_bbox_2 = pred_cell[C+5:C+9]

        IoU_1 = IoU(pred_bbox_1, target_bbox, row, col, True)
        IoU_2 = IoU(pred_bbox_2, target_bbox, row, col,  True)

        return 0 if IoU_1 > IoU_2 else 1

    def forward(self, pred, target):
        total_loss = 0
        batch_size = pred.shape[0]

        # compute losses for each image in the batch, total the losses, and then average them
        for b in range(batch_size):
            loss_1 = loss_2 = loss_3 = loss_4 = loss_5 = 0

            for i in range(S):
                for j in range(S):
                    pred_cell = pred[b][i][j]
                    target_cell = target[b][i][j]

                    if target_cell[C+4] == 1: # if cell has object
                        responsible = self._find_responsible_(pred_cell, target_cell, i, j)
                        pred_bbox = pred_cell[C:C+5] if responsible == 0 else pred_cell[C+5:C+10]
                        target_bbox = target_cell[C:C+5]

                        # 1. Sum of Squared Errors (SSE) for center points
                        loss_1 += (target_bbox[0] - pred_bbox[0]).square() + \
                                  (target_bbox[1] - pred_bbox[1]).square()

                        # 2. SSE for square roots of width and height
                        # avoid errors with negative predictions
                        pred_w = torch.sign(pred_bbox[2]) * (pred_bbox[2].abs() + 1e-6).sqrt()
                        pred_h = torch.sign(pred_bbox[3]) * (pred_bbox[3].abs() + 1e-6).sqrt()

                        loss_2 += (target_bbox[2].sqrt() - pred_w).square() + \
                                  (target_bbox[3].sqrt() - pred_h).square()

                        # 3. SSE for confidences of responsible bounding box predictors
                        loss_3 += (target_bbox[4] - pred_bbox[4]).square()

                        # 4. SSE for confidences of not responsible bounding box predictors
                        other_idx = C+4 if responsible == 1 else C+9
                        loss_4 += (target_cell[other_idx] - pred_cell[other_idx]).square()

                        # 5. SSE for class probabilities
                        for c in range(C):
                            loss_5 += (target_cell[c] - pred_cell[c]).square()

                    else:
                        # 4. SSE for confidences of cells with no objects
                        loss_4 += (target_cell[C+4] - pred_cell[C+4]).square() + \
                                  (target_cell[C+9] - pred_cell[C+9]).square()

            total_loss += (LAMBDA_COORD * (loss_1 + loss_2)) + \
                          loss_3 + (LAMBDA_NOOBJ * loss_4) + loss_5

        return total_loss / batch_size