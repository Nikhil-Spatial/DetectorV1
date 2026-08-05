from src.transforms import trainval_transforms, test_transforms
from src.inference_functions import compute_eval_stats
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from src.visualization import plot_history
from src.configs import SEED, BATCH_SIZE
from src.train_functions import train
from src.dataset import ImageDataset
from src.loss_function import Loss
from src.model import Model
from pathlib import Path
import torch

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
num_epochs = 15

model = Model().to(device)
loss_fn = Loss().to(device)

optimizer = torch.optim.SGD(model.parameters(), 1e-2, 0.9, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

# 3. Training loop
checkpoint_dir = Path("../outputs/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_loss_history, val_loss_history = [], []
train_mAP_history, val_mAP_history = [], []

for epoch in range(num_epochs):
    # a. Train Model
    train_loss = train(model, loss_fn, optimizer, train_dl, device)
    scheduler.step()

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
        "val_loss": val_loss,
        "scheduler_state_dict": scheduler.state_dict()
    }

    torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # e. Display Statistics
    print(f"(Epoch {epoch+1}) Training: Loss - {train_loss:.4f} mAP - "
          f"{train_mAP:.4f} | Validation: Loss - {val_loss} mAP - {val_mAP}")

# 4. Plot and save all the history lists
epoch_list = list(range(1, num_epochs+1))

plot_history(epoch_list, train_loss_history, "training_loss")
plot_history(epoch_list, val_loss_history, "val_loss")
plot_history(epoch_list, train_mAP_history, "training_mAP")
plot_history(epoch_list, val_mAP_history, "val_mAP")


