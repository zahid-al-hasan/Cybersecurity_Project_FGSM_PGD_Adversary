"""
Baseline CNN model for CIFAR-10 classification.
This is the target model that will be attacked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 image classification.
    Architecture:
        Conv2D(3, 32, 3) -> ReLU -> MaxPool
        Conv2D(32, 64, 3) -> ReLU -> MaxPool
        Conv2D(64, 128, 3) -> ReLU -> MaxPool
        Flatten -> FC(128*4*4, 256) -> ReLU -> Dropout
        FC(256, 10)
    """

    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()

        # TODO: Define convolutional layers
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # TODO: Define fully connected layers
        self.fc_1 = nn.Linear(in_features=128*4*4, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=10)

        # TODO: Define dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # TODO: Implement forward pass
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.flatten(x)

        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_2(x)

        return x
        # Conv layers -> Flatten -> FC layers -> Output
