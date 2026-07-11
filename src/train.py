from transforms import trainval_transforms, test_transforms
from train_functions import train, compute_accuracy
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
trainval_dataset = ImageDataset(annot_file_trainval, img_dir_trainval)
test_dataset = ImageDataset(annot_file_test, img_dir_test)

generator1 = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(trainval_dataset, [0.8, 0.2])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32)
test_dl = DataLoader(test_dataset, batch_size=32)

# 2. Instantiate model, loss, etc.
model = Model()
loss_fn = Loss()

optimizer = torch.optim.SGD(model.parameters(), 1e-2, 0.9, weight_decay=0.0005)
lr_scheduler = CosineAnnealingLR(optimizer, eta_min=1e-4)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 75

# 3. Training Loop
checkpoint_dir = Path("../outputs/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

loss_history, train_mAP_history, val_mAP_history = [], [], []

for epoch in range(epochs):
    # 1. Train Model
    loss = train(model, loss_fn, optimizer, train_dl, device)
    loss_history.append(loss)
    lr_scheduler.step()

    # 2. Evaluate on Training Data
    train_mAP, _, _ = compute_accuracy(model, train_dl, device)
    train_mAP_history.append(train_mAP)

    # 3. Evaluate on Validation Data
    val_mAP, _, _ = compute_accuracy(model, val_dl, device)
    val_mAP_history.append(val_mAP)

    # 4. Display Statistics
    print(f"Epoch {epoch+1}: Loss - {loss} | mAP on Training Set - {train_mAP} "
          f"| mAP on Validation Set - {val_mAP}")