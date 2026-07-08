import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init()

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

class ResidualBlock1(nn.Module):
    def __init__(self):
        super().__init__()

        # Sequence 1: Convolutional Layer -> BatchNorm -> ReLU
        self.conv_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(128)

        # Sequence 2: Convolutional Layer -> BatchNorm -> shortcut + add -> ReLU
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(128)

        self.shortcut = nn.Conv2d(64, 128, kernel_size=1, stride=2)

    def forward(self, x):
        original_x = x

        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))
        return F.relu(self.shortcut(original_x) + x)

class ResidualBlock2(nn.Module):
    def __init__(self):
        super().__init__()

        # Sequence 1: Convolutional Layer -> BatchNorm -> ReLU
        self.conv_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(128)

        # Sequence 2: Convolutional Layer -> BatchNorm -> add -> ReLU
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(128)

    def forward(self, x):
        original_x = x

        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))
        return F.relu(original_x + x)

class ResidualBlock3(nn.Module):
    def __init__(self):
        super().__init__()

        # Sequence 1: Convolutional Layer -> BatchNorm -> ReLU
        self.conv_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(256)

        # Sequence 2: Convolutional Layer -> BatchNorm -> shortcut + add -> ReLU
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(256)

        self.shortcut = nn.Conv2d(128, 256, kernel_size=1, stride=2)

    def forward(self, x):
        original_x = x

        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))
        return F.relu(shortcut(original_x) + x)

class ResidualBlock4(nn.Module):
    def __init__(self):
        super().__init__()

        # Sequence 1: Convolutional Layer -> BatchNorm -> ReLU
        self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(256)

        # Sequence 2: Convolutional Layer -> BatchNorm -> shortcut + add -> ReLU
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        original_x = x

        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))
        return F.relu(self.shortcut(original_x) + x)


