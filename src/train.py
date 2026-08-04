from src.transforms import trainval_transforms, test_transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader
from src.dataset import ImageDataset
from src.loss_function import Loss
from src.configs import SEED
from src.model import Model
from pathlib import Path
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

generator_ = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(trainval_dataset, [0.8, 0.2]
                                          ,generator=generator_)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32)
test_dl = DataLoader(test_dataset, batch_size=32)

# 2. Instantiate model, loss function, device, optimizer, and scheduler
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 75

model = Model().to(device)
loss_fn = Loss().to(device)

optimizer = torch.optim.SGD(model.parameters(), 1e-2, 0.9, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

# 3. Training Loop
checkpoint_dir = Path("../outputs/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_loss_history, val_loss_history = [], []
train_mAP_history, val_mAP_history = [], []

for epoch in range(num_epochs):
    # 1. Train Model


    # 2. Evaluate Model Performance on Training Dataset

    # 3. Evaluate Model Performance on Validation Dataset

    # 4. Save Checkpoints

    # 5. Display Statistics

    pass


