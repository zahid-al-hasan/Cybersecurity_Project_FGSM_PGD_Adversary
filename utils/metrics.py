"""
Evaluation metrics for measuring attack success and model robustness.
"""

import torch
import torch.nn as nn
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from config import *


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss = criterion(output, labels)

            total_loss += loss.item() * labels.size(0)
            predicted = output.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total



def evaluate_robustness(model, test_loader, device, attack_type, epsilon, alpha=None, steps=None):
    """
    Evaluate model accuracy under adversarial attack.

    Args:
        model: trained model
        test_loader: DataLoader for test set
        device: cuda or cpu
        attack_type: 'fgsm' or 'pgd'
        epsilon: perturbation magnitude
        alpha: PGD step size (only for PGD)
        steps: PGD iterations (only for PGD)

    Returns:
        accuracy (float)
    """

    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError("attack_type must be 'fgsm' or 'pgd'")

    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
    
        if attack_type == "fgsm":
            adversarial_images = fgsm_attack(images, labels, model=model, epsilon=epsilon)
        else:
            adversarial_images = pgd_attack(images, labels, model=model, epsilon=epsilon)
    
        with torch.no_grad():
            predictions = model.forward(adversarial_images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return 100 * (correct / total) if total else 0.0


def evaluate_over_epsilons(model, test_loader, device, attack_type, epsilons,
                            alpha=None, steps=None):
    """
    Evaluate model accuracy over a range of epsilon values.

    Args:
        model: trained model
        test_loader: DataLoader for test set
        device: cuda or cpu
        attack_type: 'fgsm' or 'pgd'
        epsilons: list of epsilon values
        alpha: PGD step size
        steps: PGD iterations

    Returns:
        list of accuracies, one per epsilon
    """

    # TODO: Loop over epsilons, call evaluate_robustness for each
    pass
