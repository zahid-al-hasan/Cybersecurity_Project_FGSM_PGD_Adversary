"""
Visualization utilities for plotting results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch, os
from config import *


def plot_accuracy_vs_epsilon(base_acc, epsilons, fgsm_std, fgsm_adv, pgd_std, pgd_adv):
    """
    Plot model accuracy vs epsilon for both FGSM and PGD attacks.

    Args:
        epsilons: list of epsilon values
        fgsm_accuracies: accuracy under FGSM for each epsilon
        pgd_accuracies: accuracy under PGD for each epsilon
    """
    
    # PLOTS_DIR = os.path.join(BASE_DIR, "plots")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Figure 1: FGSM Attack Performance
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, fgsm_std, 'o-', color='#e74c3c', label='Standard CNN (Under FGSM)', linewidth=2)
    plt.plot(epsilons, fgsm_adv, 's--', color='#2ecc71', label='Adversarially Trained CNN (Under FGSM)', linewidth=2)
    plt.title('FGSM Attack: Standard vs. Adversarially Trained Model Accuracy', fontsize=12)
    plt.xlabel('Epsilon Perturbation Magnitude ($\epsilon$)', fontsize=10)
    plt.ylabel('Model Accuracy (%)', fontsize=10)
    plt.ylim(-5, 100)
    plt.axhline(
            y=base_acc, 
            color='black', 
            linestyle='--', 
            linewidth=2.5, 
            label=f'Clean Baseline ({base_acc}%)'
        )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fgsm_comparison.png'))
    plt.close()

    # Figure 2: PGD Attack Performance
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, pgd_std, 'o-', color='#e74c3c', label='Standard CNN (Under PGD)', linewidth=2)
    plt.plot(epsilons, pgd_adv, 's--', color='#3498db', label='Adversarially Trained CNN (Under PGD)', linewidth=2)
    plt.title('PGD Attack: Standard vs. Adversarially Trained Model Accuracy', fontsize=12)
    plt.xlabel('Epsilon Perturbation Magnitude ($\epsilon$)', fontsize=10)
    plt.ylabel('Model Accuracy (%)', fontsize=10)
    plt.ylim(-5, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pgd_comparison.png'))
    plt.close()

    print("Plots generated successfully!")
    pass


# def plot_training_history(history):
#     """
#     Plot training history (loss and accuracy over epochs).

#     Args:
#         history: dict with keys train_loss, train_acc, test_loss, test_acc, robust_acc
#     """

#     # TODO: Plot loss curves (train vs test)
#     # TODO: Plot accuracy curves (clean vs robust)
#     pass
