from src.configs import C, TP_IOU_THRESHOLD, IDX_TO_CLASS
from operator import itemgetter
from src.utilities import IoU
import torch

def find_tp_and_fp(suppressed_preds, ground_truth_objects, tp_fp_by_class):
    """Find and store the True Positive and False Positive predictions for a
    single instance."""
    for class_name, preds in suppressed_preds.items():
        if class_name in ground_truth_objects:
            # Iterate through each prediction, find the ground truth object that
            # has the highest IoU with the prediction, and if that IoU surpasses
            # the threshold, it is a True Positive, else a False Positive.
            for pred in preds:
                print(tp_fp_by_class)

                IoUs = [IoU(ground_truth_object, pred[1:5]) for
                        ground_truth_object in ground_truth_objects[class_name]]

                max_idx = IoUs.index(max(IoUs))

                if IoUs[max_idx] < TP_IOU_THRESHOLD:
                    tp_fp_by_class[class_name].append((pred[0], False))

                # If the tensor contains all negative values, it's already been
                # associated with a prediction
                elif torch.all(ground_truth_objects[class_name][max_idx] < 0):
                    tp_fp_by_class[class_name].append((pred[0], False))

                # Everything else is a True Positive. To mark the associated
                # ground truth object as N/A, replace every value in the tensor
                # with negative values
                else:
                    tp_fp_by_class[class_name].append((pred[0], True))
                    ground_truth_objects[class_name][max_idx] = (
                        torch.arange(-5, -4, -3, -2))

        else:
            # The rest of the predictions belonging to the class are False
            # Positives since there are no ground truth objects belonging to
            # that class
            for pred in preds:
                tp_fp_by_class[class_name].append((pred[0], False))

def compute_map(all_tp_fp_by_class, class_object_totals,
                precision_recall_lists):
    ap_by_class = {}

    for class_name, tp_fp_list in all_tp_fp_by_class.items():
        if class_object_totals[class_name] == 0:
            continue

        AP = 0

        # 1. sort every list of TP/FPs in each class
        tp_fp_list.sort(key=itemgetter(0), reverse=True)

        # 2. iterate through list of TP/FPs and compute precision/recall
        true_positives = 0

        precision_denom = 0  # denominator - <total TP or FP>
        recall_denom = class_object_totals[
            class_name]  # denominator - <total objects in class>

        previous_recall = 0

        precision_list = precision_recall_lists[class_name][0]
        recall_list = precision_recall_lists[class_name][1]

        for _, status in tp_fp_list:
            true_positives += status
            precision_denom += 1

            precision = true_positives / precision_denom
            recall = true_positives / recall_denom

            precision_list.append(precision)
            recall_list.append(recall)

            delta_recall = recall - previous_recall
            AP += precision * delta_recall

            previous_recall = recall

        ap_by_class[class_name] = AP

    mAP = sum(ap_by_class.values()) / len(ap_by_class)

    return mAP, ap_by_class