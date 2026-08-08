IMAGE_SIZE = 224

# seed for reproducibility
SEED = 7

# grid dimensions 7 x 7
S = 7

# number of bounding boxes each cell predicts
B = 1

# number of classes
C = 20

# used to filter the raw predictions
CONFIDENCE_THRESHOLD = 0.375

# used to filter bounding boxes that attempt to box the same object
NMS_IOU_THRESHOLD = 0.9999

# used to find which predictions are true positives
TP_IOU_THRESHOLD = 0.5

# height and width of a single grid cell in pixels
CELL_SIZE = int(IMAGE_SIZE / S)

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

# Parameters to increase the loss from bounding box coordinate predictions and
# decrease the loss from confidence predictions for boxes that don't contain
# objects. So, these are for the loss functions.
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5

# size of batches in DataLoader
BATCH_SIZE = 32