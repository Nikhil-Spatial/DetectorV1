from torchvision.transforms import v2
import torch

# The mean and standard deviations across each channel for the normalized pixels
# of every single image in the "trainval" dataset
MEANS = (0.4485, 0.4249, 0.3922)
STDS = (0.2682, 0.2655, 0.2782)

color_transform = torch.nn.ModuleList([
    v2.ColorJitter(
        brightness=[0.6, 1.4],
        contrast=[0.6, 1.4],
        saturation=[0.6, 1.4]
    )
])

trainval_transforms = v2.Compose([
    v2.RandomApply(color_transform, p=1),
    v2.Normalize(mean=MEANS, std=STDS)
])

test_transforms = v2.Compose([
    v2.Normalize(mean=MEANS, std=STDS)
])

def revert_normalization(image):
    image *= 255
    return image.to(torch.uint8)

def revert_standardization(image):
    for i in range(3):
        image[i] *= STDS[i]
        image[i] += MEANS[i]