from train_test_functions import train, compute_loss_accuracy, plot_history
from transforms import trainval_transforms, test_transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from dataset import ImageDataset
from loss_function import Loss
from pathlib import Path
from configs import SEED
from model import Model
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
train_dataset, val_dataset = random_split(trainval_dataset, [0.8, 0.2]
                                          ,generator=generator1)

debug_train_dataset, debug_val_dataset, _ = random_split(trainval_dataset, [0.0032, 0.0032, 0.9936])

debug_train_dl = DataLoader(debug_train_dataset, batch_size=16, shuffle=True)
debug_val_dl = DataLoader(debug_val_dataset, batch_size=16, shuffle=True)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32)
test_dl = DataLoader(test_dataset, batch_size=32)

# 2. Instantiate model, loss, etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
num_epochs = 75

model = Model().to(device)
loss_fn = Loss().to(device)

optimizer = torch.optim.SGD(model.parameters(), 1e-5, 0.9, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

# 3. Training Loop
checkpoint_dir = Path("../outputs/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_loss_history, val_loss_history = [], []
train_mAP_history, val_mAP_history = [], []
train_mAP_epochs = []

for epoch in range(1, num_epochs + 1):
    # 1. Train Model
    train_loss = train(model, loss_fn, optimizer, debug_train_dl, device)
    train_loss_history.append(train_loss)
    scheduler.step()

    # 2. Evaluate on Validation Data
    val_mAP, val_loss, _, _ = compute_loss_accuracy(
        model, loss_fn, debug_val_dl, device)
    val_loss_history.append(val_loss)
    val_mAP_history.append(val_mAP)

    # train_mAP = "N/A"
    # if epoch % 5 == 0:
    #     # 3. Evaluate on Training Data Every 5 Epochs
    #     train_mAP, _, _, _ = compute_loss_accuracy(
    #         model, loss_fn, train_dl, device)
    #     train_mAP_history.append(train_mAP)
    #     train_mAP_epochs.append(epoch)
    #
    #     # 4. Save Checkpoints Every 5 Epochs
    #     checkpoint = {
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "val_loss": val_loss,
    #         "scheduler_state_dict": scheduler.state_dict()
    #     }
    #
    #     torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
    #
    #     print(f"(Epoch {epoch}) Training: Loss - {train_loss:.4f} "
    #           f"mAP - {train_mAP:.4f} | "
    #           f"Validation: Loss - {val_loss:.4f} mAP - {val_mAP:.4f}")

    train_mAP, _, _, _ = compute_loss_accuracy(model, loss_fn, debug_train_dl, device)

    # 5. Display Statistics
    print(f"(Epoch {epoch}) Training: Loss - {train_loss:.4f} mAP - {train_mAP:.4f}| "
          f"Validation: Loss - {val_loss:.4f} mAP - {val_mAP:.4f}")

# plot all the lists of histories
epoch_list = list(range(1, num_epochs + 1))

plot_history(epoch_list, train_loss_history, "training_loss")
plot_history(epoch_list, val_loss_history, "val_loss")
plot_history(train_mAP_epochs, train_mAP_history, "training_mAP")
plot_history(epoch_list, val_mAP_history, "val_mAP")