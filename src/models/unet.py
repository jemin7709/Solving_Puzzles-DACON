import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_mid = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(mid_channels)
        self.relu_mid = nn.ReLU()
        self.conv_block = ConvBlock(mid_channels + mid_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.relu_mid(self.bn_mid(self.conv_mid(x)))
        x = torch.cat((x, skip), dim=1)
        return self.conv_block(x)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Contraction path
        self.conv1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ConvBlock(128, 256)

        # Expansion path
        self.up6 = DeconvBlock(256, 128, 128)
        self.up7 = DeconvBlock(128, 64, 64)
        self.up8 = DeconvBlock(64, 32, 32)
        self.up9 = DeconvBlock(32, 16, 16)

        self.final_pool = nn.MaxPool2d(2)
        self.final_conv = nn.Conv2d(16, 16, kernel_size=28, stride=28)
        self.final_bn = nn.BatchNorm2d(16)

    def forward(self, x):
        # Contraction path
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        x3 = self.conv3(x)
        x = self.pool3(x3)
        x4 = self.conv4(x)
        x = self.pool4(x4)
        x5 = self.conv5(x)

        # Expansion path
        x = self.up6(x5, x4)
        x = self.up7(x, x3)
        x = self.up8(x, x2)
        x = self.up9(x, x1)

        x = self.final_pool(x)
        out = self.final_bn(self.final_conv(x))  # (B,16,4,4)
        return out
