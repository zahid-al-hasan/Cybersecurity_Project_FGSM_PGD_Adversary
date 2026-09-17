"""
Training script for baseline CNN on CIFAR-10.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn import BaselineCNN
from config import *


def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs):
    """
    Train the model on clean CIFAR-10 data.

    Args:
        model: neural network
        train_loader: training data loader
        test_loader: test data loader
        optimizer: optimizer
        criterion: loss function
        device: cuda or cpu
        num_epochs: number of epochs

    Returns:
        trained model, history dict
    """

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # TODO: Forward pass
            # TODO: Compute loss
            # TODO: Backward pass and optimizer step
            # TODO: Track metrics

            pbar.set_postfix({"loss": f"{total_loss/total:.4f}",
                              "acc": f"{100.*correct/total:.2f}"})

        # TODO: Evaluate on test set
        # TODO: Record history
        # TODO: Print summary

    return model, history


def main():
    # TODO: Set device
    # TODO: Load CIFAR-10 with transforms
    # TODO: Create data loaders
    # TODO: Initialize model, optimizer, criterion
    # TODO: Train model
    # TODO: Save model checkpoint
    pass


if __name__ == "__main__":
    main()
