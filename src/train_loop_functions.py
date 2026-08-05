from src.evaluation import (find_tp_and_fp, count_objects_in_each_class,
                            evaluate)
from src.utilities import (get_ground_truth_objects_by_class,
                           get_empty_precision_recall_lists,
                           get_empty_class_object_totals,
                           get_empty_tp_fp_by_class)
from src.postprocessing import postprocess_preds
from src.configs import IDX_TO_CLASS, BATCH_SIZE
from matplotlib import pyplot as plt
from pathlib import Path
import torch

def train(model, loss_fn, optimizer, train_dl, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_dl:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. forward pass
        preds_batch = model(X_batch)

        # 2. compute loss
        loss = loss_fn(preds_batch, y_batch)

        # 3. reset gradients
        optimizer.zero_grad()

        # 4. compute gradients
        loss.backward()

        # 5. optimizer step
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dl) # returns average loss

def compute_eval_stats(model, dl, device, loss_fn=None):
    if loss_fn: total_loss = 0

    tp_fp_by_class = get_empty_tp_fp_by_class()
    class_object_totals = get_empty_class_object_totals()
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
                val_loss = loss_fn(preds_batch, y_batch)
                total_val_loss += val_loss.item()

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
        mAP, ap_by_class = evaluate(tp_fp_by_class, class_object_totals,
                                    precision_recall_lists)

    if loss_fn:
        avg_val_loss = total_val_loss / len(dl)
        return mAP, ap_by_class, avg_val_loss, precision_recall_lists

    return mAP, ap_by_class, precision_recall_lists
