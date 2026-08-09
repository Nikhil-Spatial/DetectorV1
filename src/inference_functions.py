from evaluation import (find_tp_and_fp, count_objects_in_each_class,
                        evaluate)
from utilities import (get_ground_truth_objects_by_class,
                       get_empty_class_object_totals,
                       get_empty_tp_fp_by_class)
from postprocessing import postprocess_preds
import torch

def compute_eval_stats(model, dl, device, loss_fn=None, loss_only=False,
                       test=False):
    if loss_fn:
        total_loss = 0

    if not loss_only:
        tp_fp_by_class = get_empty_tp_fp_by_class()
        class_object_totals = get_empty_class_object_totals()

    model.eval()
    with (torch.no_grad()):
        for X_batch, y_batch in dl:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # 1. forward pass
            preds_batch = model(X_batch)

            # 2. compute validation loss (optional)
            if loss_fn:
                loss = loss_fn(preds_batch, y_batch)
                total_loss += loss.item()

            if not loss_only:
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
                    count_objects_in_each_class(
                        ground_truth_objects_by_class, class_object_totals
                    )

        # 4. Compute average precision (AP) for each class and mean average
        # precision (mAP) across all classes
        if not loss_only:
            if test:
                mAP, ap_by_class = evaluate(
                    tp_fp_by_class, class_object_totals, True
                )
            else:
                mAP = evaluate(tp_fp_by_class, class_object_totals)

    if loss_fn:
        avg_loss = total_loss / len(dl)

        if loss_only:
            return avg_loss

        return mAP, avg_loss

    elif test:
        return mAP, ap_by_class

    return mAP