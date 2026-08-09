from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet18
import torch.nn.functional as F
from src.configs import S, B, C
import torch.nn as nn
import torch

class DetectorHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv= nn.Conv2d(512, (C + B * 5), 1)

    def forward(self, x):
        return self.conv(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # use resnet18 as the feature extractor/pretrained backbone
        resnet = resnet18(weights="DEFAULT")
        resnet_feature_extractor = create_feature_extractor(resnet, ["layer4"])

        self.backbone = resnet_feature_extractor
        self.dropout = nn.Dropout()
        self.detector_head = DetectorHead()

    def forward(self, x):
        x = self.backbone(x)["layer4"]
        x = self.dropout(x)
        x = self.detector_head(x)

        x = x.permute(0, 2, 3, 1)

        # Use sigmoid activation function on the x, y, w, h, and confidence
        for item in range(x.shape[0]):
            for i in range(S):
                for j in range(S):
                    x[item][i][j][C:C+2] = F.sigmoid(x[item][i][j][C:C+2])
                    x[item][i][j][C+4] = F.sigmoid(x[item][i][j][C+4])

        return x