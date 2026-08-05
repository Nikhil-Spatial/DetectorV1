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


