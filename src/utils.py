from src.configs import CELL_SIZE, IMAGE_SIZE
from torch import round as rd
import torch

def convert_xywh_coords(bbox: tuple, row, col, draw: bool):
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
    return (xmin, ymin, xmax, ymax)

def area(bbox: tuple):
    w = torch.clamp(bbox[2] - bbox[0], min=0)
    h = torch.clamp(bbox[3] - bbox[1], min=0)
    return w * h

def IoU(target_bbox: tuple, pred_bbox: tuple, row, col):
    # 1. convert (x, y, w, h) to (xmin, ymin, xmax, ymax)
    target = convert_xywh_coords(target_bbox, row, col, False)
    pred = convert_xywh_coords(pred_bbox, row, col, False)

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