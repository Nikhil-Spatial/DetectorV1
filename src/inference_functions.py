from src.evaluation import (find_tp_and_fp, count_objects_in_each_class,
                            evaluate)
from src.utilities import (get_ground_truth_objects_by_class,
                           get_empty_precision_recall_lists,
                           get_empty_class_object_totals,
                           get_empty_tp_fp_by_class)
from src.postprocessing import postprocess_preds
import torch

def compute_eval_stats(model, dl, device, loss_fn=None, test=False):
    if loss_fn:
        total_loss = 0

    tp_fp_by_class = get_empty_tp_fp_by_class()
    class_object_totals = get_empty_class_object_totals()
    if test:
        precision_recall_lists = get_empty_precision_recall_lists()

    model.eval()
    with (torch.no_grad()):
        for X_batch, y_batch in dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 1. forward pass
            preds_batch = model(X_batch)

            # 2. compute validation loss (optional)
            if loss_fn:
                loss = loss_fn(preds_batch, y_batch)
                total_loss += loss.item()

            # 3. a) postprocess predictions, b) sort and bucket ground truth
            # objects by class, c) find True and False Positives, and d) count
            # the number of objects in each class
            for preds, target in zip(preds_batch, y_batch):
                postprocessed_preds = postprocess_preds(preds)
                ground_truth_objects_by_class = (
                    get_ground_truth_objects_by_class(target)
                )

                find_tp_and_fp(postprocessed_preds,
                               ground_truth_objects_by_class, tp_fp_by_class)
                count_objects_in_each_class(ground_truth_objects_by_class,
                                            class_object_totals)

        # 4. Compute average precision (AP) for each class and mean average
        # precision (mAP) across all classes
        if test:
            mAP, ap_by_class = evaluate(tp_fp_by_class, class_object_totals,
                                        precision_recall_lists, True)
        else:
            mAP = evaluate(tp_fp_by_class, class_object_totals)

    if loss_fn:
        avg_loss = total_loss / len(dl)
        return mAP, avg_loss

    elif test:
        return mAP, ap_by_class, precision_recall_lists

    return mAP