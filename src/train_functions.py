from evaluation import (compute_map, find_objects_in_target,
                        tp_fp_and_count_objects)
from postprocessing import nms

def train(model, loss_fn, optimizer, train_dl, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_dl:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Forward Pass
        preds = model(X_batch)

        # 2. Compute Loss
        loss = loss_fn(preds, y_batch)

        # 3. Reset Gradients
        optimizer.zero_grad()

        # 4. Compute Gradients
        loss.backward()

        # 5. Optimizer Step
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dl) # returns average loss

def compute_accuracy(model, dl, device):
    model.eval()
    # every class is associated with a tuple that contains two lists
    # list 1 contains precision scores, and list 2 recall
    precision_recall_lists = {
        "aeroplane": ([], []),
        "bicycle": ([], []),
        "bird": ([], []),
        "boat": ([], []),
        "bottle": ([], []),
        "bus": ([], []),
        "car": ([], []),
        "cat": ([], []),
        "chair": ([], []),
        "cow": ([], []),
        "diningtable": ([], []),
        "dog": ([], []),
        "horse": ([], []),
        "motorbike": ([], []),
        "person": ([], []),
        "pottedplant": ([], []),
        "sheep": ([], []),
        "sofa": ([], []),
        "train": ([], []),
        "tvmonitor": ([], []),
    }

    # every class is associated with a list of tuples
    # each tuple is of the form (confidence, bool)
    # if bool is True, then TP, and if bool is False, then FP
    all_tp_fp_by_class = {
        "aeroplane": [],
        "bicycle": [],
        "bird": [],
        "boat": [],
        "bottle": [],
        "bus": [],
        "car": [],
        "cat": [],
        "chair": [],
        "cow": [],
        "diningtable": [],
        "dog": [],
        "horse": [],
        "motorbike": [],
        "person": [],
        "pottedplant": [],
        "sheep": [],
        "sofa": [],
        "train": [],
        "tvmonitor": [],
    }

    # every class is associated with how many times they appear as objects in
    # the entire dataset
    class_object_totals = {
        "aeroplane": 0,
        "bicycle": 0,
        "bird": 0,
        "boat": 0,
        "bottle": 0,
        "bus": 0,
        "car": 0,
        "cat": 0,
        "chair": 0,
        "cow": 0,
        "diningtable": 0,
        "dog": 0,
        "horse": 0,
        "motorbike": 0,
        "person": 0,
        "pottedplant": 0,
        "sheep": 0,
        "sofa": 0,
        "train": 0,
        "tvmonitor": 0,
    }

    for X_batch, y_batch in dl:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Forward Pass
        preds = model(X_batch)

        # 2. Postprocess
        final_preds = nms(preds)
        ground_truth_objects = find_objects_in_target(y_batch, device)

        # 3. Find TP/FP and count total objects per class
        tp_fp_and_count_objects(final_preds, ground_truth_objects,
                                all_tp_fp_by_class, class_object_totals)

    mAP, ap_by_class = compute_map(all_tp_fp_by_class, class_object_totals,
                                   precision_recall_lists)

    return mAP, ap_by_class, precision_recall_lists