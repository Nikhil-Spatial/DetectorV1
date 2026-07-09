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

    return total_loss / len(train_dl)

def compute_accuracy