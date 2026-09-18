"""
Visualization utilities for plotting results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_examples(original, adversarial, labels, predictions, num_examples=5,
                  save_path=None):
    """
    Plot original vs adversarial images side by side.

    Args:
        original: batch of clean images
        adversarial: batch of adversarial images
        labels: true labels
        predictions: model predictions on adversarial images
        num_examples: how many images to show
    """

    count = min(num_examples, original.size(0))
    figure, axes = plt.subplots(2, count, figsize=(3 * count, 6), squeeze=False)

    for index in range(count):
        clean_image = original[index].detach().cpu().permute(1, 2, 0).numpy()
        adversarial_image = adversarial[index].detach().cpu().permute(1, 2, 0).numpy()
        axes[0, index].imshow(np.clip(clean_image, 0, 1))
        axes[0, index].set_title(f"True: {int(labels[index])}")
        axes[1, index].imshow(np.clip(adversarial_image, 0, 1))
        axes[1, index].set_title(f"Pred: {int(predictions[index])}")
        axes[0, index].axis("off")
        axes[1, index].axis("off")

    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_accuracy_vs_epsilon(epsilons, fgsm_accuracies, pgd_accuracies,
                             save_path=None):
    """
    Plot model accuracy vs epsilon for both FGSM and PGD attacks.

    Args:
        epsilons: list of epsilon values
        fgsm_accuracies: accuracy under FGSM for each epsilon
        pgd_accuracies: accuracy under PGD for each epsilon
    """

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epsilons, fgsm_accuracies, marker="o", label="FGSM")
    axis.plot(epsilons, pgd_accuracies, marker="o", label="PGD")
    axis.set_xlabel("Epsilon")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy vs. Perturbation Budget")
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, bbox_inches="tight")
    return figure


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy over epochs).

    Args:
        history: dict with keys train_loss, train_acc, test_loss, test_acc, robust_acc
    """

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history.get("train_loss", []), label="Train")
    axes[0].plot(epochs, history.get("test_loss", []), label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.get("train_acc", []), label="Train")
    axes[1].plot(epochs, history.get("test_acc", []), label="Test")
    if history.get("robust_acc"):
        axes[1].plot(epochs, history["robust_acc"], label="Robust")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, bbox_inches="tight")
    return figure
