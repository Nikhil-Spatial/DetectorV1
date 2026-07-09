from transforms import trainval_transforms, test_transforms
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2
from pathlib import Path
import torch

annot_file_trainval = Path("../data/preprocessed/trainval/annotations.csv")
img_dir_trainval = Path("../data/preprocessed/trainval/Images")

annot_file_test = Path("../data/preprocessed/test/annotations.csv")
img_dir_test = Path("../data/preprocessed/test/Images")

# 1. Datasets and DataLoaders




