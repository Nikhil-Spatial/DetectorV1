import torch
from model import Model
from pathlib import Path
from dataset import ImageDataset
from transforms import test_transforms
from configs import BATCH_SIZE
from torch.utils.data import DataLoader
from inference_functions import compute_eval_stats

device = "cuda" if torch.cuda.is_available() else "cpu"

# load best checkpoint
best_checkpoint_path = Path("../outputs/checkpoints/checkpoint_epoch_70.pth")
best_checkpoint = torch.load(best_checkpoint_path)

# load parameters into the model
model = Model().to(device, non_blocking=True)
model.load_state_dict(best_checkpoint["model_state_dict"])

# load test dataset and create dataloader
annot_file_test = Path("../data/preprocessed/test/annotations.csv")
img_dir_test = Path("../data/preprocessed/test/Images")

test_dataset = ImageDataset(annot_file_test, img_dir_test,
                            transform=test_transforms)

test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4,
                     pin_memory=True, persistent_workers=True)

# perform inference
mAP, ap_by_class, precision_recall_lists = compute_eval_stats(
    model, test_dl, device, test=True
)

# print stats and plot precision
