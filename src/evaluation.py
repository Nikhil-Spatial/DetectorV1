from src.utilities import IoU
from src.configs import C, TP_IOU_THRESHOLD, IDX_TO_CLASS
from operator import itemgetter
import torch

def find_tp_and_fp(postprocessed_preds, ground_truth_objects_by_class,
                   tp_fp_by_class):
    """Find and store the True Positive and False Positive predictions for a
    single instance."""
    for class_name, preds in postprocessed_preds.items():
        if class_name in ground_truth_objects_by_class:
            # Iterate through each prediction, find the ground truth object that
            # has the highest IoU with the prediction, and if that IoU surpasses
            # the threshold, it is a True Positive, else a False Positive.
            for pred in preds:
                IoUs = [IoU(ground_truth_object, pred[1:5]) for
                        ground_truth_object in
                        ground_truth_objects_by_class[class_name]]

                max_idx = IoUs.index(max(IoUs))

                if IoUs[max_idx] < TP_IOU_THRESHOLD:
                    tp_fp_by_class[class_name].append((pred[0], False))

                # If the tensor contains all negative values, it's already been
                # associated with a prediction
                elif all(x < 0 for x in
                         ground_truth_objects_by_class[class_name][max_idx]):
                    tp_fp_by_class[class_name].append((pred[0], False))

                # Everything else is a True Positive. To mark the associated
                # ground truth object as N/A, replace every value in the tensor
                # with negative values
                else:
                    tp_fp_by_class[class_name].append((pred[0], True))
                    ground_truth_objects_by_class[class_name][max_idx] = (
                        torch.arange(-5, -1))

        else:
            # The rest of the predictions belonging to the class are False
            # Positives since there are no ground truth objects belonging to
            # that class
            for pred in preds:
                tp_fp_by_class[class_name].append((pred[0], False))

def count_objects_in_each_class(ground_truth_objects_by_class,
                                class_object_totals):
    """Add the number of objects that belong to each class for one image to the
    total in the dictionary."""
    for class_name, ground_truth_objects in ground_truth_objects_by_class.items():
        class_object_totals[class_name] += len(ground_truth_objects)

def average_precision(tp_fp_by_class, class_object_totals,
                      precision_recall_lists=None):
    ap_by_class = {}

    for class_name, tp_fp_list in tp_fp_by_class.items():
        if class_object_totals[class_name] == 0:
            continue

        # declare/reset average precision to zero
        AP = 0

        # 1. sort every list of TP/FPs in each class by descending order
        tp_fp_list.sort(key=itemgetter(0), reverse=True)

        # 2. iterate through list of TP/FPs and compute precision/recall
        true_positives = 0

        precision_denom = 0  # denominator: <Total TP + FP>
        recall_denom = class_object_totals[class_name]  # denominator - <total objects in class>

        previous_recall = 0

        if precision_recall_lists:
            precision_list = precision_recall_lists[class_name][0]
            recall_list = precision_recall_lists[class_name][1]

        # iteratively, compute the area under the precision-recall curve via
        # Riemann sums
        for _, status in tp_fp_list:
            true_positives += status
            precision_denom += 1

            precision = true_positives / precision_denom
            recall = true_positives / recall_denom

            if precision_recall_lists:
                precision_list.append(precision)
                recall_list.append(recall)

            delta_recall = recall - previous_recall
            AP += (precision * delta_recall)

            previous_recall = recall

        ap_by_class[class_name] = AP

    return ap_by_class

def mean_average_precision(ap_by_class):
    mAP = sum(ap_by_class.values()) / len(ap_by_class)

    return mAP

def evaluate(tp_fp_by_class, class_object_totals, precision_recall_lists=None,
             ap_by_class_return=False):
    # 1. compute average precision (AP) for each class
    ap_by_class = average_precision(tp_fp_by_class, class_object_totals,
                                    precision_recall_lists)

    # 2. compute mean average precision (mAP)
    mAP = mean_average_precision(ap_by_class)

    if ap_by_class_return:
        return mAP, ap_by_class

    return mAP