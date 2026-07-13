from src.configs import LAMBDA_COORD, LAMBDA_NOOBJ, S, C, B
import torch

class Loss(torch.nn.Module):
    """The loss function from the YOLOv1 paper."""
    def __init__(self):
        super().__init__()

    def _compute_loss_1(self, pred_bbox, target_bbox):
        """Sum of Squared Errors (SSE) for center points"""
        loss = (target_bbox[0] - pred_bbox[0]).square() + \
            (target_bbox[1] - pred_bbox[1]).square()

        return LAMBDA_COORD * loss

    def _compute_loss_2(self, pred_bbox, target_bbox):
        """SSE for square roots of width and height"""
        # Avoid errors caused by taking the square root of predictions that are
        # either negative or zero
        pred_w = torch.sign(pred_bbox[2]) * (pred_bbox[2].abs() + 1e-8).sqrt()
        pred_h = torch.sign(pred_bbox[3]) * (pred_bbox[3].abs() + 1e-8).sqrt()

        loss = (target_bbox[2].sqrt() - pred_w).square() + \
            (target_bbox[3].sqrt() - pred_h).square()

        return LAMBDA_COORD * loss

    def _compute_loss_3(self, pred_confidence, target_confidence):
        """SSE for confidences of bounding box predictors"""
        loss = (target_confidence - pred_confidence).square()

        return loss

    def _compute_loss_4(self, pred_confidence, target_confidence):
        """SSE for confidences of cells with no objects"""
        loss = (target_confidence - pred_confidence).square()

        return LAMBDA_NOOBJ * loss

    def _compute_loss_5(self, pred_cell, target_cell):
        """SSE for class probabilities"""
        loss = 0
        for c in range(C):
            loss += (target_cell[c] - pred_cell[c]).square()

        return loss

    def forward(self, pred, target):
        total_loss = 0
        batch_size = pred.shape[0]

        # For each image in the batch, 1) compute losses, 2) total the losses,
        # and then 3) average them
        for b in range(batch_size):
            loss_1 = loss_2 = loss_3 = loss_4 = loss_5 = 0

            for i in range(S):
                for j in range(S):
                    pred_cell = pred[b][i][j]
                    target_cell = target[b][i][j]

                    pred_confidence = pred_cell[C + 4]
                    target_confidence = target_cell[C + 4]

                    # If the cell contains an object, compute losses 1-3, 5
                    if target_confidence == 1:
                        pred_bbox = pred_cell[C:C + 4]
                        target_bbox = target_cell[C:C+4]

                        loss_1 += self._compute_loss_1(pred_bbox, target_bbox)
                        loss_2 += self._compute_loss_2(pred_bbox, target_bbox)
                        loss_3 += self._compute_loss_3(pred_confidence,
                                                      target_confidence)
                        loss_5 += self._compute_loss_5(pred_cell, target_cell)

                    # If the cell does not contain an object, compute loss 4
                    else:
                        loss_4 += self._compute_loss_4(pred_confidence,
                                                     target_confidence)

            total_loss += (loss_1 + loss_2 + loss_3 + loss_4 + loss_5)

        average_loss = total_loss / batch_size

        return average_loss