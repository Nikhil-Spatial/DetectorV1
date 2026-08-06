from configs import S, B, C
import torch.nn.functional as F
import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # Sequence 1: Convolutional Layer -> BatchNorm -> ReLU
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        # Sequence 2: Convolutional Layer -> BatchNorm -> shortcut + add -> ReLU
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        # check if shortcut is required
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        original_x = x

        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))

        return F.relu(self.shortcut(original_x) + x)

class ResidualBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer1
        self.conv_1 = nn.Conv2d(3, 64, 7, padding=3)

        # Residual Blocks
        self.res_blocks = nn.ModuleList([])

        self.res_blocks.append(ResidualBlock(64, 128, 2))
        self.res_blocks.append(ResidualBlock(128, 128, 1))
        self.res_blocks.append(ResidualBlock(128, 128, 1))
        self.res_blocks.append(ResidualBlock(128, 256, 2))
        self.res_blocks.append(ResidualBlock(256, 256, 2))
        self.res_blocks.append(ResidualBlock(256, 256, 1))
        self.res_blocks.append(ResidualBlock(256, 256, 1))
        self.res_blocks.append(ResidualBlock(256, 512, 2))
        self.res_blocks.append(ResidualBlock(512, 512, 2))

    def forward(self, x):
        x = F.relu(self.conv_1(x))

        for res_block in self.res_blocks:
            x = res_block(x)

        return x

class DetectorHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv= nn.Conv2d(512, (C + B * 5), 1)

    def forward(self, x):
        return self.conv(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ResidualBackbone()
        self.detector_head = DetectorHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector_head(x)

        x = x.reshape((x.shape[0], S, S, C + B * 5))

        # Use sigmoid activation function on the x, y, w, h, and confidence
        for item in range(x.shape[0]):
            for i in range(7):
                for j in range(7):
                    x[item][i][j][20:25] = F.sigmoid(x[item][i][j][20:25])

        return x