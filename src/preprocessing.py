import xml.etree.ElementTree as ET
from configs import IMAGE_SIZE
from pathlib import Path
from PIL import Image
import csv

# parse individual xml files
def parse_xml(xml_path, writer):
    tree = ET.parse(xml_path)
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
        x = int(round((xmin+xmax)/2))
        y = int(round((ymin+ymax)/2))
        w = int(round(xmax-xmin))
        h = int(round(ymax-ymin))

        # write to annotations.csv file
        writer.writerow([filename, class_name, x, y, w, h])

# write to csv annotations files
def write_csv(csv_annot_dir, annot_dir):
    csv_annot_dir.mkdir(parents=True, exist_ok=True)

    with open (csv_annot_dir / "annotations.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        writer.writerow(["filename", "class_name", "x", "y", "w", "h"])

        for annot_path in annot_dir.glob("*.xml"):
            parse_xml(annot_path, writer)

# resize image dimensions to 224x224
def process_images(img_dir, preprocessed_img_dir):
    preprocessed_img_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_dir.glob("*.jpg"):
        preprocessed_img_path = preprocessed_img_dir / img_path.name

        with Image.open(img_path) as img:
            processed_img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            processed_img.save(preprocessed_img_path)

# 1) raw annotations -> structure annotations.csv file
# 2) raw image dimensions -> resize to 224x224
def preprocess():
    # store the train/val and test annotations directories
    trainval_annot_dir = Path("../data/raw/VOCtrainval-2007/Annotations")
    test_annot_dir = Path("../data/raw/VOCtest-2007/Annotations")

    trainval_csv_annot_dir = Path("../data/preprocessed/trainval")
    test_csv_annot_dir = Path("../data/preprocessed/test")

    write_csv(trainval_csv_annot_dir, trainval_annot_dir)
    write_csv(test_csv_annot_dir, test_annot_dir)

    # store the train/val and test image directories---raw and preprocessed
    trainval_img_dir = Path("../data/raw/VOCtrainval-2007/JPEGImages")
    test_img_dir = Path("../data/raw/VOCtest-2007/JPEGImages")

    trainval_preprocessed_img_dir = Path("../data/preprocessed/trainval/Images")
    test_preprocessed_img_dir = Path("../data/preprocessed/test/Images")

    process_images(trainval_img_dir, trainval_preprocessed_img_dir)
    process_images(test_img_dir, test_preprocessed_img_dir)

if __name__ == "__main__":
    preprocess()