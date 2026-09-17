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
        # TODO: Define fully connected layers
        # TODO: Define dropout

    def forward(self, x):
        # TODO: Implement forward pass
        # Conv layers -> Flatten -> FC layers -> Output
        pass
