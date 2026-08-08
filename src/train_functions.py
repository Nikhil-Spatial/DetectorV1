def train(model, loss_fn, optimizer, train_dl, device):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_dl:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

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

        epoch_loss += loss.item()

    return epoch_loss / len(train_dl) # returns average loss over the epoch


