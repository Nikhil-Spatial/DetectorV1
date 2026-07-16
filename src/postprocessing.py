from src.configs import (S, C, B, IDX_TO_CLASS, CONFIDENCE_THRESHOLD,
                         NMS_IOU_THRESHOLD)
from src.utilities import convert_xywh_coordinates, IoU
from operator import itemgetter
import torch

def decode_preds(preds):
    # Stores prediction class name, confidence score, and bbox of one instance
    decoded_preds = []

    for i in range(S):
        for j in range(S):
            cell = preds[i][j]

            class_idx = cell[:C].argmax()
            class_prob = cell[class_idx]

            confidence_score = cell[C+4] * class_prob
            bbox = convert_xywh_coordinates(cell[C:C+4], i, j, False)

            decoded_preds.append((class_idx, confidence_score) + bbox)

    return decoded_preds

def filter_and_group_preds(decoded_preds):
    """Filter predictions with confidence scores less than the threshold, and
    group every remaining prediction by class name."""
    filtered_grouped_preds = {}

    for pred in decoded_preds:
        if pred[1] > CONFIDENCE_THRESHOLD:
            class_name = IDX_TO_CLASS[pred[0].item()]

            if class_name in filtered_grouped_preds:
                filtered_grouped_preds[class_name].append(pred[1:])

            else:
                filtered_grouped_preds[class_name] = [pred[1:]]

    return filtered_grouped_preds

def sort_class_preds_by_confidence(filtered_grouped_preds):
    """Sort the predictions in every class of the filtered and grouped
    predictions dictionary by confidence score in increasing order."""
    for class_name in filtered_grouped_preds:
        filtered_grouped_preds[class_name].sort(key=itemgetter(0))

def non_maximum_suppression(filtered_grouped_preds):
    """Perform non-maximum suppression to remove predictions that attempt to
    bound the same ground truth object."""
    suppressed_preds = {}

    for class_name, preds in filtered_grouped_preds.items():
        suppressed_preds[class_name] = []

        while preds: # while preds is not empty
            # 1. Pop and store prediction with the highest confidence
            highest_conf = preds.pop()
            suppressed_preds[class_name].append(highest_conf)

            print(highest_conf)
            for pred in preds:
                print(pred)
                print(IoU(highest_conf[1:5], pred[1:5]))

            # 2. Filter out or "suppress" the bboxes that are too similar
            preds = [pred for pred in preds if
                     IoU(highest_conf[1:5], pred[1:5]) < NMS_IOU_THRESHOLD]

    # Just a note, the predictions under each class are now sorted in decreasing
    # order instead of increasing order.
    return suppressed_preds