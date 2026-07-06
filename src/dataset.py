import xml.etree.ElementTree as ET
from configs import IMAGE_SIZE
from pathlib import Path
import numpy as np
import os

# store the annotations directory
annot_dir = Path("../data/VOCtrainval-2007/Annotations")
annot_list = os.listdir(annot_dir)

# parse individual xml files
def parse_xml(xml_path):
    tree = ET.parse(os.path.join(annot_dir, xml_path))
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # store pixel ratios to scale coordinates accordingly
    img_width_ratio = IMAGE_SIZE / width
    img_height_ratio = IMAGE_SIZE / height

    objects = []
    for obj in root.findall("object"):
        class_name = obj.find



