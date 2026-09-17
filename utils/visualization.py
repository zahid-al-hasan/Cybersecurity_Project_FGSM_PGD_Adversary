"""
Visualization utilities for plotting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_examples(original, adversarial, labels, predictions, num_examples=5):
    """
    Plot original vs adversarial images side by side.

    Args:
        original: batch of clean images
        adversarial: batch of adversarial images
        labels: true labels
        predictions: model predictions on adversarial images
        num_examples: how many images to show
    """

    # TODO: Create a grid of plots
    # - Top row: original images with true labels
    # - Bottom row: adversarial images with predicted labels
    # - Show the perturbation (difference) as well
    pass


def plot_accuracy_vs_epsilon(epsilons, fgsm_accuracies, pgd_accuracies):
    """
    Plot model accuracy vs epsilon for both FGSM and PGD attacks.

    Args:
        epsilons: list of epsilon values
        fgsm_accuracies: accuracy under FGSM for each epsilon
        pgd_accuracies: accuracy under PGD for each epsilon
    """

    # TODO: Plot line chart with epsilon on x-axis, accuracy on y-axis
    # Two lines: one for FGSM, one for PGD
    pass


def plot_training_history(history):
    """
    Plot training history (loss and accuracy over epochs).

    Args:
        history: dict with keys train_loss, train_acc, test_loss, test_acc, robust_acc
    """

    # TODO: Plot loss curves (train vs test)
    # TODO: Plot accuracy curves (clean vs robust)
    pass
