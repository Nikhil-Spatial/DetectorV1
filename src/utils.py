from src.configs import CELL_SIZE, IMAGE_SIZE, S, B, C
from torch import round as rd
import torch

def convert_xywh_coords(bbox, row, col, draw: bool, tensor: bool):
    # 1. unnormalize
    x = (bbox[0] * CELL_SIZE) + (col * CELL_SIZE)
    y = (bbox[1] * CELL_SIZE) + (row * CELL_SIZE)
    w = bbox[2] * IMAGE_SIZE
    h = bbox[3] * IMAGE_SIZE

    # 2. convert center point to top-left and bottom-right of box coordinates
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2

    # 3. if the conversion is for drawing bounding boxes, then round
    if draw:
        return (int(rd(xmin)), int(rd(ymin)), int(rd(xmax)), int(rd(ymax)))

    if tensor:
        return (xmin.item(), ymin.item(), xmax.item(), ymax.item())
    else:
        return (xmin, ymin, xmax, ymax)


def area(bbox):
    w = torch.clamp(bbox[2] - bbox[0], min=0)
    h = torch.clamp(bbox[3] - bbox[1], min=0)
    return w * h

def IoU(pred, target, row, col, conversion_needed: bool):
    # 1. convert (x, y, w, h) to (xmin, ymin, xmax, ymax) --- if needed
    if conversion_needed:
        target = convert_xywh_coords(target, row, col, False)
        pred = convert_xywh_coords(pred, row, col, False)

    # 2. find intersection box coordinates
    xmin = torch.max(target[0], pred[0])
    ymin = torch.max(target[1], pred[1])
    xmax = torch.min(target[2], pred[2])
    ymax = torch.min(target[3], pred[3])

    inter = (xmin, ymin, xmax, ymax)

    # 3. find areas of boxes
    target_area = area(target)
    pred_area = area(pred)
    inter_area = area(inter)
    union_area = target_area + pred_area - inter_area

    # 4. compute IoU
    return inter_area / union_area if union_area != 0 else 0

def decode_preds(preds_batch):
    decoded_preds = []

    for pred in preds_batch:
        pred = pred.reshape((S, S, C + B * 5))
        objects = []

        for i in range(S):
            for j in range(S):
                pred_cell = pred[i][j]

                bbox_1 = convert_xywh_coords(pred_cell[20:24], i, j, False, True)
                bbox_2 = convert_xywh_coords(pred_cell[25:29], i, j, False, True)

                pred_1_confidence = (pred_cell[24].item(),)
                pred_2_confidence = (pred_cell[29].item(),)

                objects.append(bbox_1 + pred_1_confidence)
                objects.append(bbox_2 + pred_2_confidence)

        decoded_preds.append(objects)

    return decoded_preds