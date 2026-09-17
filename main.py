"""
Main entry point - runs the full pipeline:
    Phase 1: Train baseline model
    Phase 2: Attack with FGSM and PGD
    Phase 3: Evaluate and compare results
    (Optional) Phase 4: Adversarial training and re-evaluation

Usage:
    python main.py
"""

import os
import torch
from config import *
from models.cnn import BaselineCNN
from data.cifar10_parquet import get_dataloaders
from train import train
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from utils.metrics import evaluate_model, evaluate_robustness, evaluate_over_epsilons
from utils.visualization import plot_accuracy_vs_epsilon


def phase1_train_baseline():
    """
    Phase 1: Train baseline CNN on CIFAR-10.
    Save checkpoint to checkpoints/baseline.pth
    """
    train_loader, test_loader = get_dataloaders(PARQUET_DATA_ROOT, BATCH_SIZE)
    model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model, history = train(train_loader, test_loader, model=model, device=DEVICE)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "baseline.pth"))
    return model


def phase2_attack_baseline():
    """
    Phase 2: Load baseline model, attack with FGSM and PGD.
    Save results to results/
    """
    # TODO: Load baseline model
    # TODO: Run FGSM over range of epsilons
    # TODO: Run PGD over range of epsilons
    # TODO: Save accuracy vs epsilon plots
    pass


def phase3_defense():
    """
    Phase 3 (Optional): Adversarial training + re-evaluate.
    Save checkpoint to checkpoints/adv_trained.pth
    """
    # TODO: Adversarial training
    # TODO: Evaluate robust model against FGSM and PGD
    # TODO: Compare with baseline
    pass


def main():
    print("=" * 60)
    print("FGSM/PGD Adversarial Examples on CIFAR-10 Image Classifier")
    print("=" * 60)

    print("\n--- Phase 1: Training Baseline Model ---")
    phase1_train_baseline()

    print("\n--- Phase 2: Attacking Baseline Model ---")
    phase2_attack_baseline()

    print("\n--- Phase 3: Defense (Adversarial Training) ---")
    phase3_defense()

    print("\nDone! Check results/ for plots and analysis.")


if __name__ == "__main__":
    main()
