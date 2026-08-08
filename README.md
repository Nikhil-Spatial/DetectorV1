# 📷 **DetectorV1**

As my first computer vision project, I decided to loosely implement YOLOv1 to gain experience with object detectors.
This is not a complete paper implementation, however, because I changed aspects of the original paper such as the architecture and preprocessing to better accommodate my hardware restrictions. 
I developed this project from scratch, writing code for data preprocessing and postprocessing, loss function, evaluation functions, and the training loop, among other things. 

## 🔄 Preprocessing
- I converted the bounding box coordinates of ground truth labels from (x1, y1, x2, y2), where each x-y pair represents a corner of the bounding box, to (x, y, w, h) where x and y represent the center of the bounding box, and w and h represent the width and height of the bounding box, respectively.
- Due to memory and GPU restrictions, I resized every image to 224 x 224 instead of 448 x 448, as described in the YOLOv1 paper. The biggest benefit of this was considerably speeding up training time.
- I normalized the image pixels to map to [0, 1] by dividing each pixel by 255.
- I standardized the image pixels using the mean and standard deviations of the ImageNet-1k dataset (see *Model Architecture*).

## 👷 Model Architecture 
- The backbone is the ResNet-18 model pretrained on the ImageNet-1k dataset. The model's classification head was cut off, specifically the global average pooling and fully connected layers.
- The detector head is a simple convolutional layer with a kernel size of 1 that outputs 25 channels.
- In the 25 channels, the first 20 channels represent the classification portion of the task, and the final 5 channels represent the bounding box prediction (x, y, w, h, and confidence score).
- The model's output vector is of the form [B, W, H, C], where B = batch size, W and H are the spatial dimensions, and C is the channel count.
- In the bounding box portion of the prediction vector, the x, y, and confidence score predictions are activated using a sigmoid layer to map, potentially, negative values to [0, 1].
