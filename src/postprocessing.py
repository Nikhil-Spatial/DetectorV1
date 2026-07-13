from src.configs import (S, C, B, IDX_TO_CLASS, CONFIDENCE_THRESHOLD,
                         NMS_IOU_THRESHOLD)
from src.utilities import convert_xywh_coordinates, IoU
from operator import itemgetter
import torch

def decode_preds_batch(preds_batch):
    decoded_preds_batch = []

    for preds in preds_batch:
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

        decoded_preds_batch.append(decoded_preds)

    return torch.tensor(decoded_preds_batch)

def filter_and_group_preds(decoded_preds):
    """Filter predictions with confidence scores less than the threshold, and
    group every remaining prediction by class name."""
    filtered_grouped_preds = {}

    for pred in decoded_preds:
        if pred[1] > CONFIDENCE_THRESHOLD:
            class_name = IDX_TO_CLASS[pred[0].item()]

            if class_name in filtered_grouped_preds:
                filtered_grouped_preds[class_name].append(pred)

            else:
                filtered_grouped_preds[class_name] = [pred]

    return filtered_grouped_preds

def sort_class_preds_by_confidence(filtered_grouped_preds):
    """Sort the predictions in every class of the filtered and grouped
    predictions dictionary by confidence score in ascending order."""
    for class_name in filtered_grouped_preds:
        filtered_grouped_preds[class_name].sort(key=itemgetter(1))

def nms(preds_batch):
    # 1. decode batch of predictions
    decoded_preds = decode_preds(preds_batch)

    # 2. filter, group, and sort the decoded predictions
    sorted_preds = filter_group_sort_preds(decoded_preds)

    # 3. perform Non-Maximum Suppression
    final_preds = []

    for image in sorted_preds:
        final_img_preds = {}

        for class_name, preds in image.items():
            final_img_preds[class_name] = []

            while preds:
                highest_conf = preds.pop()
                final_img_preds[class_name].append(highest_conf)

                preds = [pred for pred in preds if
                         IoU(highest_conf[2:6], pred[2:6]) < NMS_IOU_THRESHOLD]

        final_preds.append(final_img_preds)

    return final_preds