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
import json
import torch
import torch.optim as optim
from config import *
from models.cnn import BaselineCNN
from dataset_loader import get_dataloaders
from train import train
from utils.metrics import evaluate_model, evaluate_robustness, evaluate_over_epsilons
from utils.visualization import plot_accuracy_vs_epsilon, plot_training_history
from defenses.adversarial_training import adversarial_training


EPSILONS = [0.0, 0.01, 0.02, 0.03]


def _ensure_output_directories():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def phase1_train_baseline():
    """
    Phase 1: Train baseline CNN on CIFAR-10.
    Save checkpoint to checkpoints/baseline.pth
    """
    _ensure_output_directories()
    train_loader, test_loader = get_dataloaders(PARQUET_DATA_ROOT, BATCH_SIZE)
    model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model, history = train(train_loader, test_loader, model=model, device=DEVICE)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "baseline.pth"))
    plot_training_history(history, os.path.join(RESULTS_DIR, "baseline_training.png"))
    return model, history


def phase2_attack_baseline():
    """
    Phase 2: Load baseline model, attack with FGSM and PGD.
    Save results to results/
    """
    _ensure_output_directories()
    _, test_loader = get_dataloaders(PARQUET_DATA_ROOT, BATCH_SIZE)
    model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "baseline.pth"), map_location=DEVICE
    ))
    results = {
        "epsilons": EPSILONS,
        "clean_accuracy": evaluate_model(model, test_loader, DEVICE),
        "fgsm_accuracies": evaluate_over_epsilons(
            model, test_loader, DEVICE, "fgsm", EPSILONS
        ),
        "pgd_accuracies": evaluate_over_epsilons(
            model, test_loader, DEVICE, "pgd", EPSILONS,
            alpha=PGD_ALPHA, steps=PGD_STEPS
        ),
    }
    plot_accuracy_vs_epsilon(
        EPSILONS,
        results["fgsm_accuracies"],
        results["pgd_accuracies"],
        os.path.join(RESULTS_DIR, "baseline_accuracy_vs_epsilon.png"),
    )
    with open(os.path.join(RESULTS_DIR, "baseline_evaluation.json"), "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    return results


def phase3_defense():
    """
    Phase 3 (Optional): Adversarial training + re-evaluate.
    Save checkpoint to checkpoints/adv_trained.pth
    """
    _ensure_output_directories()
    train_loader, test_loader = get_dataloaders(PARQUET_DATA_ROOT, BATCH_SIZE)
    model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "baseline.pth"), map_location=DEVICE
    ))
    baseline_results = {
        "clean_accuracy": evaluate_model(model, test_loader, DEVICE),
        "fgsm_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "fgsm", FGSM_EPSILON
        ),
        "pgd_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "pgd", PGD_EPSILON,
            alpha=PGD_ALPHA, steps=PGD_STEPS
        ),
    }
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model, history = adversarial_training(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler=None,
        device=DEVICE,
        epsilon=ADV_TRAIN_EPSILON,
        alpha=PGD_ALPHA,
        pgd_steps=PGD_STEPS,
        num_epochs=ADV_TRAIN_EPOCHS,
    )
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "adv_trained.pth"))
    plot_training_history(history, os.path.join(RESULTS_DIR, "adv_training.png"))
    results = {
        "clean_accuracy": evaluate_model(model, test_loader, DEVICE),
        "fgsm_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "fgsm", FGSM_EPSILON
        ),
        "pgd_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "pgd", PGD_EPSILON,
            alpha=PGD_ALPHA, steps=PGD_STEPS
        ),
    }
    with open(os.path.join(RESULTS_DIR, "adv_trained_evaluation.json"), "w", encoding="utf-8") as output_file:
        json.dump({"baseline": baseline_results, "adversarially_trained": results}, output_file, indent=2)
    return model, history, results


def main():
    set_seed()
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
