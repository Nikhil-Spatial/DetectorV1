# 📷 **DetectorV1**

As my first computer vision project, I decided to loosely implement YOLOv1 to gain experience with object detectors.
This is not a complete paper implementation, however, because I changed aspects of the original paper such as the architecture and preprocessing to better accommodate my hardware restrictions. 
I developed this project from scratch, writing code for data preprocessing and postprocessing, loss function, evaluation functions, and the training loop, among other things. 

## 📌 Methods

Preprocessing: 
- I converted the bounding box coordinates of ground truth labels from (x1, y1, x2, y2), where each x-y pair represents a corner of the bounding box, to (x, y, w, h) where x and y represent the center of the bounding box, and w and h represent the width and height of the bounding box, respectively.
- Due to memory and GPU restrictions, I resized every image to 224 x 224 instead of 448 x 448, as described in the YOLOv1 paper. The biggest benefit of this was considerably speeding up training time.
- I normalized the image pixels to fall in [0, 1] by dividing each pixel by 255.
- I standardized the image pixels using the mean and standard deviations of the ImageNet-1k dataset because the model's backbone is pretrained on ImageNet-1k.
