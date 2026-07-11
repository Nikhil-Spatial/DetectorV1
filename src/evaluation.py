from configs import C, TP_IOU_THRESHOLD
from utils import IoU
import torch

def find_objects_in_target(y_batch, device):
    y_batch = y_batch.flatten(1, 2)
    truth_objects = []

    for target in y_batch:
        class_objects = {}

        for cell in target:
            class_name = IDX_TO_CLASS[int(torch.argmax(cell[:C]))]

            if cell[C + 4] == 1:
                # the additional 0 is to classify that specific object as
                # unmatched with a prediction. 1 is for matched.
                bboxes = torch.cat(
                    (cell[C:C + 4], torch.tensor([0]).to(device)))
                if class_name in class_objects:
                    class_objects[class_name].append(bboxes)
                else:
                    class_objects[class_name] = [bboxes]

        truth_objects.append(class_objects)

    return truth_objects

def tp_fp_and_count_objects(final_preds, truth_objects, all_tp_fp_by_class,
                            class_object_totals):
    # 1. iterate through each image prediction/label in the batch
    for b in range(len(final_preds)):
        img_preds = final_preds[b]
        objects = truth_objects[b]

        # 2. iterate through each class
        for class_name, preds in img_preds.items():
            # a. if any of the ground truth objects belong to the class
            if class_name in objects:
                # iterate through each prediction, find max IoU truth object, and
                # if the max IoU surpasses the threshold, it is a TP, else FP
                for pred in preds:
                    IoUs = [IoU(object_[0:4], pred[2:6]) for object_ in
                            objects[class_name]]
                    max_idx = IoUs.index(max(IoUs))

                    if IoUs[max_idx] < TP_IOU_THRESHOLD:
                        all_tp_fp_by_class[class_name].append((pred[1], False))

                    elif objects[class_name][max_idx][-1] == 0:
                        all_tp_fp_by_class[class_name].append((pred[1], True))
                        objects[class_name][max_idx][-1] = 1

                    else:
                        all_tp_fp_by_class[class_name].append((pred[1], False))
            else:
                # b. all predictions belonging to class are FP since there are
                # no ground truth objects belonging to that class
                for pred in preds:
                    all_tp_fp_by_class[class_name].append((pred[1], False))

        # 3. add the number of objects that belong to each class to the total
        for class_name, object_list in objects.items():
            class_object_totals[class_name] += len(object_list)

def mean_average_precision(all_tp_fp_by_class, class_object_totals,
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