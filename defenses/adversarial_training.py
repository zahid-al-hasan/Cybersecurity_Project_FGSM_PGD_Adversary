"""
Adversarial Training Defense Implementation.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.

Adversarial training augments the training data with adversarial examples
generated during each epoch, forcing the model to learn robust features.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from attacks.pgd import pgd_attack
from utils.metrics import evaluate_robustness


def adversarial_training(model, train_loader, test_loader, optimizer, scheduler,
                         device, epsilon, alpha, pgd_steps, num_epochs):
    """
    Train a model with PGD adversarial examples.

    For each batch:
        1. Generate PGD adversarial examples from the batch
        2. Compute loss on adversarial examples
        3. Update model weights

    Args:
        model: neural network to train
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        optimizer: optimizer (e.g., Adam)
        scheduler: learning rate scheduler
        device: "cuda" or "cpu"
        epsilon: PGD epsilon
        alpha: PGD step size
        pgd_steps: number of PGD iterations
        num_epochs: number of training epochs

    Returns:
        trained model, history dict with train/test losses and accuracies
    """

    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
               "robust_acc": []}
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        batch_iterator = tqdm(
            train_loader,
            desc=f"Adv epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for images, labels in batch_iterator:
            images, labels = images.to(device), labels.to(device)

            adv_images = pgd_attack(
                images,
                labels,
                model=model,
                criterion=criterion,
                epsilon=epsilon,
                alpha=alpha,
                steps=pgd_steps,
            )
            outputs = model(adv_images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            batch_iterator.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = _evaluate_loss_and_accuracy(
            model, test_loader, criterion, device
        )
        robust_acc = 100.0 * evaluate_robustness(
            model,
            test_loader,
            device,
            attack_type="pgd",
            epsilon=epsilon,
            alpha=alpha,
            steps=pgd_steps,
        )
        print(
            f"  Epoch {epoch + 1}/{num_epochs} evaluation: "
            f"clean={test_acc:.2f}%, robust={robust_acc:.2f}%"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["robust_acc"].append(robust_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
            f"Robust Acc: {robust_acc:.2f}%"
        )

        if scheduler:
            scheduler.step()

    return model, history


def _evaluate_loss_and_accuracy(model, data_loader, criterion, device):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
    finally:
        model.train(was_training)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total
