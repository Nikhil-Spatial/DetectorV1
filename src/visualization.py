from matplotlib import pyplot as plt
import bbox_visualizer as bbv
from pathlib import Path
import math

def draw_rectangles(image, coords, labels):
    image = image.permute(1, 2, 0).numpy()
    image = bbv.draw_multiple_rectangles(image, coords, bbox_color=(255, 0, 0),
                                         thickness=1)
    image = bbv.add_multiple_labels(image, labels, coords, size=0.5,
                                    thickness=1, draw_bg=False, text_color=
                                    (255, 255, 255))

    return image