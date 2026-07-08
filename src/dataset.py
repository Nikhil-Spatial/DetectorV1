from src.configs import CLASS_TO_IDX, IDX_TO_CLASS, CELL_SIZE, IMAGE_SIZE
from torchvision.io import decode_image
from torch.utils.data import Dataset
import pandas as pd
import torch

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

    def _get_objects(self, filename):
        group = self.groups.get_group(filename).drop(columns=["filename"])

        objects = []
        for _, row in group.iterrows():
            objects.append(tuple(row))

        return objects

    def _create_target_vector(self, objects):
        target_vector = torch.zeros(7, 7, 30)
        for object_ in objects:
            row = object_[2] // CELL_SIZE  # row number of the grid cell
            col = object_[1] // CELL_SIZE  # column number of the grid cell
            cell = target_vector[row][col]

            if not cell.any():  # if another object hasn't already occupied the cell
                # one-hot encode the class label
                cell[CLASS_TO_IDX[object_[0]]] = 1

                # parameterize bbox x and y to be offsets of cell location
                cell[20] = cell[25] = (object_[1] - (col * CELL_SIZE)) / CELL_SIZE
                cell[21] = cell[26] = (object_[2] - (row * CELL_SIZE)) / CELL_SIZE

                # normalize bbox width and height
                cell[22] = cell[27] = object_[3] / IMAGE_SIZE
                cell[23] = cell[28] = object_[4] / IMAGE_SIZE

                # confidence = 1 for cells containing objects
                cell[24] = cell[29] = 1

        return target_vector

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]

        img_path = self.img_dir / img_filename
        image = decode_image(img_path)

        objects = self._get_objects(img_filename)
        target_vector = self._create_target_vector(objects)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target_vector = self.target_transform(target_vector)

        return image, target_vector