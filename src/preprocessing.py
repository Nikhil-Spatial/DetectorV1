import xml.etree.ElementTree as ET
from configs import IMAGE_SIZE
from pathlib import Path
import numpy as np
import csv
import os

# store the annotations directory
annot_dir = Path("../data/raw/VOCtrainval-2007/Annotations")
annot_list = os.listdir(annot_dir)

csv_file_path = Path("../data/preprocessed/annotations.csv")

# parse individual xml files
def parse_xml(xml_path, writer):
    tree = ET.parse(os.path.join(annot_dir, xml_path))
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # store ratios to scale bounding box coordinates
    img_width_ratio = IMAGE_SIZE / width
    img_height_ratio = IMAGE_SIZE / height

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # parse bounding box coordinates and scale them
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text) * img_width_ratio
        ymin = int(bndbox.find("ymin").text) * img_height_ratio
        xmax = int(bndbox.find("xmax").text) * img_width_ratio
        ymax = int(bndbox.find("ymax").text) * img_height_ratio

        # convert (xmin, ymin, xmax, ymax) to (x, y, w, h) format
        x = np.round(np.mean([xmin,xmax]))
        y = np.round(np.mean([ymin, ymax]))
        w = np.round((xmax - xmin) / IMAGE_SIZE)
        h = np.round((ymax - ymin) / IMAGE_SIZE)

        # write to annotations.csv file
        writer.writerow([filename, class_name, x, y, w, h])


