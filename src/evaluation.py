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