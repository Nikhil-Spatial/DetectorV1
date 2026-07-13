from src.configs import (S, C, B, IDX_TO_CLASS, CONFIDENCE_THRESHOLD,
                         NMS_IOU_THRESHOLD)
from src.utilities import convert_xywh_coords, IoU
from operator import itemgetter

def decode_preds(preds_batch):
    decoded_preds = []

    for pred in preds_batch:
        pred = pred.reshape((S, S, C + B * 5))
        objects = []

        for i in range(S):
            for j in range(S):
                pred_cell = pred[i][j]

                pred_class_idx = pred_cell[:C].argmax().item()
                pred_class_prob = pred_cell[pred_class_idx].item()
                pred_class = IDX_TO_CLASS[pred_class_idx]

                pred_1_confidence = pred_cell[C+4].item() * pred_class_prob
                pred_2_confidence = pred_cell[C+9].item() * pred_class_prob

                bbox_1 = convert_xywh_coords(pred_cell[C:C+4], i, j, False)
                bbox_2 = convert_xywh_coords(pred_cell[C+5:C+9], i, j, False)

                objects.append((pred_class, pred_1_confidence) + bbox_1)
                objects.append((pred_class, pred_2_confidence) + bbox_2)

        decoded_preds.append(objects)

    return decoded_preds

def filter_group_sort_preds(decoded_preds):
    sorted_preds = []

    # 1. filter and group remaining predictions by class
    for image in decoded_preds:
        valid_preds = {}
        for pred in image:
            if pred[1] > CONFIDENCE_THRESHOLD:
                class_name = pred[0]
                if class_name in valid_preds:
                    valid_preds[class_name].append(pred)
                else:
                    valid_preds[class_name] = [pred]

        sorted_preds.append(valid_preds)

    # 2. sort each class's predictions by confidence score
    for image in sorted_preds:
        for class_name in image:
            image[class_name].sort(key=itemgetter(1))

    return sorted_preds

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