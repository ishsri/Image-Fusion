from turtle import forward
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNET (nn.Module):
    def __init__( self, in_channels=3, out_channels=1, features=[ 64, 128, 256, 512], 

    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET architecture
        for feature in features:
            self.downs.append(double_conv(in_channels, feature))
            in_channels = feature

        #Up part of UNET architecture
        for feature in features:
            self.ups.append(double_conv(out_channels, feature))
            out_channels = feature