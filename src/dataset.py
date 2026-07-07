from torchvision.io import decode_image
from torch.utils.data import Dataset
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
        return len(self.filenames)

    def _get_labels(self, filename):
        group = self.groups.get_group(filename).drop(columns=["filename"])

        objects = []
        for i in group.iterrows():
            objects.append(tuple(i[1]))

        return objects

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]

        img_path = self.img_dir / img_filename
        image = decode_image(img_path)

        labels = self._get_labels(img_filename)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels