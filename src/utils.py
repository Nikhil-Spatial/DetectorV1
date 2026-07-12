from src.configs import CELL_SIZE, IMAGE_SIZE, S, B, C, IDX_TO_CLASS
from torch import round as rd
import torch

def convert_xywh_coords(bbox, row, col, draw: bool):
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

def area(bbox):
    w = torch.clamp(bbox[2] - bbox[0], min=0)
    h = torch.clamp(bbox[3] - bbox[1], min=0)
    return w * h

def IoU(bbox_1, bbox_2, row=None, col=None, conversion_needed=False):
    # 1. convert (x, y, w, h) to (xmin, ymin, xmax, ymax) --- if needed
    if conversion_needed:
        bbox_1 = convert_xywh_coords(bbox_1, row, col, False)
        bbox_2 = convert_xywh_coords(bbox_2, row, col, False)

    # 2. find intersection box coordinates
    xmin = torch.max(bbox_1[0], bbox_2[0])
    ymin = torch.max(bbox_1[1], bbox_2[1])
    xmax = torch.min(bbox_1[2], bbox_2[2])
    ymax = torch.min(bbox_1[3], bbox_2[3])

    inter = (xmin, ymin, xmax, ymax)

    # 3. find areas of boxes
    target_area = area(bbox_1)
    pred_area = area(bbox_2)
    inter_area = area(inter)
    union_area = target_area + pred_area - inter_area

    # 4. compute IoU
    return inter_area / union_area if union_area != 0 else 0

def find_objects_in_target(y_batch, nms: bool):
    truth_objects_by_img = []
    batch_size = y_batch.shape[0]

    for b in range(batch_size):
        objects_by_class = {}

        for i in range(S):
            for j in range(S):
                cell = y_batch[b][i][j]

                class_idx = int(torch.argmax(cell[:C]))
                class_name = IDX_TO_CLASS[class_idx]

                # if the cell contains the center of an object
                if cell[C + 4] == 1:
                    # convert targets from (x, y, w, h) to (x1, y1, x2, y2)
                    bbox = convert_xywh_coords(cell[C:C + 4], i, j, True) # CHANGED DRAW TO TRUE FOR TESTING PURPOSES *** CHANGE THIS ***

                    # the additional 0 is to classify that specific object as
                    # unmatched with a prediction for NMS. 1 is for matched.
                    if nms:
                        bbox = torch.cat((bbox, torch.tensor([0])))

                    if class_name in objects_by_class:
                        objects_by_class[class_name].append(bbox)
                    else:
                        objects_by_class[class_name] = [bbox]

        truth_objects_by_img.append(objects_by_class)

    return truth_objects_by_img