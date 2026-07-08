IMAGE_SIZE = 224

# grid dimensions 7 x 7
S = 7

# number of bounding boxes each cell predicts
B = 2

# number of classes
C = 20

CLASS_TO_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}

IDX_TO_CLASS = {
    idx: class_
    for class_, idx in CLASS_TO_IDX.items()
}
