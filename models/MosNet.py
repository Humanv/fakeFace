# import torch
import torch.nn as nn


class MosNet(nn.Module):
    def __init__(self):
        super(MosNet, self).__init__()
        # size: 256*256*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # size:128*128*8
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # size:64*64*8
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # size:32*32*16
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        # size:8*8*16
        self.full1 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(1024, 16)
        )
        self.full2 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(16, 2)
        )
        self.end = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.full1(x)
        x = self.full2(x)
        x = self.end(x)
        return x
