"""
Evaluation metrics for measuring attack success and model robustness.
"""

import torch
import torch.nn as nn
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

    # TODO: Set model to eval mode
    # TODO: Loop through test data, compute predictions
    # TODO: Calculate and return accuracy
    pass


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

    # TODO: Set model to eval mode
    # TODO: For each batch, generate adversarial examples
    # TODO: Evaluate model on adversarial examples
    # TODO: Calculate and return robust accuracy
    pass


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
