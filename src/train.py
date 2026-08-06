from transforms import trainval_transforms, test_transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from inference_functions import compute_eval_stats
from visualization import plot_history
from configs import SEED, BATCH_SIZE
from train_functions import train
from dataset import ImageDataset
from loss_function import Loss
from pathlib import Path
from model import Model
import argparse
import torch

# add command line argument to resume training at a certain checkpoint
parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from."
)
args = parser.parse_args()

annot_file_trainval = Path("../data/preprocessed/trainval/annotations.csv")
img_dir_trainval = Path("../data/preprocessed/trainval/Images")

# 1. Datasets and DataLoaders
trainval_dataset = ImageDataset(annot_file_trainval, img_dir_trainval,
                                transform=trainval_transforms)

generator_ = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(trainval_dataset, [0.8, 0.2]
                                          ,generator=generator_)

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 2. Instantiate model, loss function, device, optimizer, and scheduler
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(device)
loss_fn = Loss().to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

start_epoch = 0
num_epochs = 15

if args.resume is not None:
    checkpoint = torch.load(args.resume)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]

    print(f"Resuming from epoch {start_epoch}")

# 3. Training loop
checkpoint_dir = Path("../outputs/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_loss_history, val_loss_history = [], []
train_mAP_history, val_mAP_history = [], []

for epoch in range(start_epoch, num_epochs):
    print(f"Starting Epoch {epoch+1}")

    # a. Train Model
    train_loss = train(model, loss_fn, optimizer, train_dl, device)

    # b. Evaluate Model Performance on Training Dataset
    train_mAP = compute_eval_stats(model, train_dl, device)
    train_loss_history.append(train_loss)
    train_mAP_history.append(train_mAP)

    # c. Evaluate Model Performance on Validation Dataset
    val_mAP, val_loss = compute_eval_stats(model, val_dl, device, loss_fn)
    val_loss_history.append(val_loss)
    val_mAP_history.append(val_mAP)

    # d. Save Checkpoints
    checkpoint = {
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # e. Display Statistics
    print(f"(Epoch {epoch+1}) Training: Loss - {train_loss:.4f} mAP - "
          f"{train_mAP:.4f} | Validation: Loss - {val_loss} mAP - {val_mAP}")

# 4. Plot and save all the history lists
epoch_list = list(range(start_epoch+1, num_epochs+1))

plot_history(epoch_list, train_loss_history, "training_loss")
plot_history(epoch_list, val_loss_history, "val_loss")
plot_history(epoch_list, train_mAP_history, "training_mAP")
plot_history(epoch_list, val_mAP_history, "val_mAP")


