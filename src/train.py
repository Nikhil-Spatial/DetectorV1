from train_test_functions import train, compute_loss_accuracy, plot_history
from transforms import trainval_transforms, test_transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from dataset import ImageDataset
from loss_fn.py import Loss
from model.py import Model
from pathlib import Path
from configs import SEED
import torch

annot_file_trainval = Path("../data/preprocessed/trainval/annotations.csv")
img_dir_trainval = Path("../data/preprocessed/trainval/Images")

annot_file_test = Path("../data/preprocessed/test/annotations.csv")
img_dir_test = Path("../data/preprocessed/test/Images")

# 1. Datasets and DataLoaders
trainval_dataset = ImageDataset(annot_file_trainval, img_dir_trainval,
                                transform=trainval_transforms)
test_dataset = ImageDataset(annot_file_test, img_dir_test,
                            transform=test_transforms)

generator1 = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(trainval_dataset, [0.8, 0.2])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32)
test_dl = DataLoader(test_dataset, batch_size=32)

# 2. Instantiate model, loss, etc.
model = Model()
loss_fn = Loss()

optimizer = torch.optim.SGD(model.parameters(), 1e-2, 0.9, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, eta_min=1e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 75

# 3. Training Loop
checkpoint_dir = Path("../outputs/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_loss_history, val_loss_history = [], []
train_mAP_history, val_mAP_history = [], []

for epoch in range(1, epochs + 1):
    # 1. Train Model
    train_loss = train(model, loss_fn, optimizer, train_dl, device)
    train_loss_history.append(loss)
    scheduler.step()

    # 2. Evaluate on Training Data
    train_mAP, _, _, _ = compute_accuracy(model, train_dl, device)
    train_mAP_history.append(train_mAP)

    # 3. Evaluate on Validation Data
    val_mAP, val_loss, _, _ = compute_accuracy(model, val_dl, device)
    val_loss_history.append(val_loss)
    val_mAP_history.append(val_mAP)

    # 4. Display Statistics
    print(f"(Epoch {epoch}) Training: Loss - {train_loss} mAP - {train_mAP} | "
          f"Validation: Loss - {val_loss} mAP - {val_mAP}")

    if epoch % 5 == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "scheduler_state_dict": scheduler.state_dict()
        }

        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

# plot all the lists of histories
plot_history(epochs, train_loss_history, "training_loss")
plot_history(epochs, val_loss_history, "val_loss")
plot_history(epochs, train_mAP_history, "training_mAP")
plot_history(epochs, val_mAP_history, "val_mAP")