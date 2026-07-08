import torch.nn.functional as F
import torch.nn as nn
import torch
from configs import S, B, C

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
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        original_x = x

        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))

        return F.relu(self.shortcut(original_x) + x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
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

        # Detector Head
        self.fc_1 = nn.Linear(25_088, 4096)
        self.fc_2 = nn.Linear(4096, (C + B * 5) * S * S)

    def forward(self, x):
        x = self.conv_1(x)
        for res_block in self.res_blocks:
            x = res_block(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)