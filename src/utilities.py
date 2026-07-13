from src.configs import IDX_TO_CLASS, CELL_SIZE, IMAGE_SIZE, S, C
from torch import round as rd
import torch

def convert_xywh_coordinates(bbox, row, col, draw: bool):
    # 1. unnormalize
    x = (bbox[0] * CELL_SIZE) + (col * CELL_SIZE)
    y = (bbox[1] * CELL_SIZE) + (row * CELL_SIZE)
    w = bbox[2] * IMAGE_SIZE
    h = bbox[3] * IMAGE_SIZE

    # 2. convert center point to top-left and bottom-right of box coordinates
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # 3. if the conversion is for drawing bounding boxes, then round
    if draw:
        return (int(rd(x1)), int(rd(y1)), int(rd(x2)), int(rd(y2)))

    return (x1, y1, x2, y2)

def area(bbox):
    w = torch.clamp(bbox[2] - bbox[0], min=0)
    h = torch.clamp(bbox[3] - bbox[1], min=0)
    return w * h

def IoU(bbox_1, bbox_2):
    # 1. find intersection box coordinates
    x1 = torch.max(bbox_1[0], bbox_2[0])
    y1 = torch.max(bbox_1[1], bbox_2[1])
    x2 = torch.min(bbox_1[2], bbox_2[2])
    y2 = torch.min(bbox_1[3], bbox_2[3])

    inter = (x1, y1, x2, y2)

    # 2. find areas of boxes
    target_area = area(bbox_1)
    pred_area = area(bbox_2)
    inter_area = area(inter)
    union_area = target_area + pred_area - inter_area

    # 3. compute IoU
    return inter_area / union_area if union_area != 0 else 0

def objects_in_target(target):
    """Returns a dictionary that associates a list of the ground truth object
    labels with each class that's present in a single image."""
    objects_by_class = {}

    for i in range(S):
        for j in range(S):
            cell = target[i][j]

            if cell[C + 4] == 1:
                class_idx = int(torch.argmax(cell[:C]))
                class_name = IDX_TO_CLASS[class_idx]
                bbox = cell[C:C+4]

                if class_name in objects_by_class:
                    objects_by_class[class_name].append(bbox)

                else:
                    objects_by_class[class_name] = [bbox]

    return objects_by_class