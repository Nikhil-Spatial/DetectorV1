from configs import LAMBDA_COORD, LAMBDA_NOOBJ, S, B
import torch.nn as nn
from uilts import IoU

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_1 = CenterLoss()
        self.loss_fn_2 = WidthHeightLoss()
        self.loss_fn_3 = ObjectConfidenceLoss()
        self.loss_fn_4 = NoObjectConfidenceLoss()
        self.loss_fn_5 = ClassificationLoss()

    def _find_responsible_(self, pred_cell, target_cell, row, col):
        """Returns 0 if the first bounding box predictor is responsible for
        a prediction, and 1 if the second is responsible."""
        target_bbox = target_cell[20:24]
        pred_bbox_1 = pred_cell[20:24]
        pred_bbox_2 = pred_cell[25:29]

        IoU_1 = IoU(pred_bbox_1, target_bbox, row, col)
        IoU_2 = IoU(pred_bbox_2, target_bbox, row, col)

        return 0 if IoU_1 > IoU_2 else 1

    def forward(self, pred, target):
        loss_1 = loss_2 = loss_3 = loss_4 = loss_5 = 0
        for i in list(range(7)):
            for j in list(range(7)):
                pred_cell = pred[i][j]
                target_cell = target[i][j]
                cell_has_object = target_cell[24] == 1 or target_cell[29] == 1

                if cell_has_object:
                    responsible = _find_responsible_(pred_cell, target_cell, i, j)
                    pred_bbox = pred_cell[20:25] if responsible == 0 else pred_cell[25:30]
                    target_bbox = target_cell[20:25]

                    # 1. Sum of Squared Errors (SSE) for center points
                    loss_1 += (target_bbox[0] - pred_bbox[0]).square() + \
                              (target_bbox[1] - pred_bbox[1]).square()

                    # 2. SSE for square roots of width and height
                    # avoid errors with negative predictions
                    pred_w = pred_bbox[2].abs().sqrt() + 1e-6
                    pred_h = pred_bbox[3].abs().sqrt() + 1e-6

                    loss_2 += (target_bbox[2].sqrt() - pred_w).square() + \
                              (target_bbox[3].sqrt() - pred_h).square()

                    # 3. SSE for confidences of responsible bounding box predictors
                    loss_3 += (target_bbox[4] - pred_bbox[4]).square()

                    # 4. SSE for confidences of not responsible bounding box predictors
                    not_responsible_idx = (responsible + 1) % 2

                    # 5. SSE for class probabilities
                    for c in list(range(20)):
                        loss_5 += (target_cell[c] - pred_cell[c]).square()

                else:
                    # 4. SSE for confidences of cells with no objects
                    loss_4 += (target_cell[24] - pred_cell[24]).square() + \
                              (target_cell[29] - pred_cell[29]).square()

        return (LAMBDA_COORD * (loss_1 + loss_2)) + loss_3 + (LAMBDA_NOOBJ * loss_4) + loss_5



