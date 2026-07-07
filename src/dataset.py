from torchvision.io import decode_image
from torch.utils.data import Dataset
from utils.py import get_labels
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        img_df = pd.read_csv(annotations_file)
        self.filenames = list(img_df["filename"].unique()) # list of image file names
        self.groups = img_df.groupby("filename") # filename groups contain respective objects

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]

        img_path = img_dir / img_filename
        image = decode_image(img_path)

        labels = get_labels(self.groups, img_filename)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels