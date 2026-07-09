from torchvision.transforms import v2

TRAINVAL_MEANS = (0.4485, 0.4249, 0.3922)
TRAINVAL_STDS = (0.2682, 0.2655, 0.2782)

color_transform = nn.ModuleList([
    v2.ColorJitter(
        brightness=[0.3, 0.7],
        contrast=[0.3, 0.7],
        saturation=[0.3, 0.7]
    )
])

trainval_transforms = v2.Compose([
    v2.RandomApply(color_transform, p=0.75),
    v2.Normalize(mean=TRAINVAL_MEANS, std=TRAINVAL_STDS)
])

test_transforms = v2.Compose([
    v2.Normalize(mean=TRAINVAL_MEANS, std=TRAINVAL_STDS)
])