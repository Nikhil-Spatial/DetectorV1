from src.configs import CELL_SIZE, IMAGE_SIZE
from torch import round as rd

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