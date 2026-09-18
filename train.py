"""
Training script for baseline CNN on CIFAR-10.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.cnn import BaselineCNN
from dataset_loader import get_dataloaders
from config import *


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item() * labels.size(0)
            predicted = output.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train(train_loader, test_loader, model=None, optimizer=None,
          criterion=None, device=DEVICE, num_epochs=NUM_EPOCHS):

    if model is None:
        model = BaselineCNN(num_classes=NUM_CLASSES)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predicted = output.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "acc": f"{100.*correct/total:.2f}"
            })

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

    return model, history


def main():
    set_seed()
    train_loader, test_loader = get_dataloaders(PARQUET_DATA_ROOT, BATCH_SIZE)
    model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model, history = train(train_loader, test_loader, model=model)
    torch.save(model.state_dict(), "checkpoints/baseline.pth")
    print("Model saved to checkpoints/baseline.pth")


if __name__ == "__main__":
    main()