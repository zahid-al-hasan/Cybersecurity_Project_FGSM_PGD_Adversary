"""
Evaluation metrics for measuring attack success and model robustness.
"""

import torch
import torch.nn as nn
from config import PGD_ALPHA, PGD_STEPS
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def evaluate_model(model, test_loader, device):
    """
    Evaluate model accuracy on clean test data.

    Args:
        model: trained model
        test_loader: DataLoader for test set
        device: cuda or cpu

    Returns:
        accuracy (float)
    """

    was_training = model.training
    model.eval()
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
    finally:
        model.train(was_training)

    return correct / total if total else 0.0


def evaluate_robustness(model, test_loader, device, attack_type, epsilon,
                         alpha=None, steps=None):
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

    was_training = model.training
    model.eval()
    correct = 0
    total = 0

    try:
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if attack_type == "fgsm":
                adversarial_images = fgsm_attack(
                    images, labels, model=model, epsilon=epsilon
                )
            else:
                adversarial_images = pgd_attack(
                    images,
                    labels,
                    model=model,
                    epsilon=epsilon,
                    alpha=PGD_ALPHA if alpha is None else alpha,
                    steps=PGD_STEPS if steps is None else steps,
                )

            with torch.no_grad():
                predictions = model(adversarial_images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    finally:
        model.train(was_training)

    return correct / total if total else 0.0


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

    return [
        evaluate_robustness(
            model,
            test_loader,
            device,
            attack_type,
            epsilon,
            alpha=alpha,
            steps=steps,
        )
        for epsilon in epsilons
    ]
