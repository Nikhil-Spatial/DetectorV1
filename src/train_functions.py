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

    for X_batch, y_batch in train_dl:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Forward Pass
        preds = model(X_batch)

        # 2. Postprocessing
        final_preds = nms(preds)
        ground_truth_objects = find_objects_in_target(y_batch, device)

        # 3. Evaluation
        mAP, ap_by_class = map(final_preds, ground_truth_objects, )