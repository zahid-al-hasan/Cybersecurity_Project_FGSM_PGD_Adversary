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
import torch, json
from torch.optim import Adam
import numpy as np
from config import *
from models.cnn import BaselineCNN
from dataset_loader import get_dataloaders
from train import train
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from defenses.adversarial_training import adversarial_training
from utils.metrics import evaluate, evaluate_robustness, evaluate_over_epsilons
from utils.visualization import plot_accuracy_vs_epsilon


train_loader, test_loader = get_dataloaders(PARQUET_DATA_ROOT, BATCH_SIZE)

def phase1_train_baseline():
    """
    Phase 1: Train baseline CNN on CIFAR-10.
    Save checkpoint to checkpoints/baseline.pth
    """

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
    model = BaselineCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "baseline.pth"), map_location=DEVICE))
    model.eval()

    epsilons = [0.005, 0.008, 0.01, 0.02, 0.05, 0.075]
    fgsm_accuracies = []
    pgd_accuracies = []

    # TODO: Run FGSM over range of epsilons
    print(f"{'-'*4} FGSM {'-'*4}")
    for eps in epsilons:
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            adv_images = fgsm_attack(images, labels, model=model, epsilon=eps)

            with torch.no_grad():
                prediction = model.forward(adv_images).argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * (correct / total)
        fgsm_accuracies.append(accuracy)
        print(f"FGSM eps={eps:.3f} | Robust Acc: {accuracy:.2f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save({"epsilons": epsilons, "fgsm_accuracies": fgsm_accuracies}, os.path.join(RESULTS_DIR, "fgsm_results.pth"))


    # TODO: Run PGD over range of epsilons
    print(f"\n{'-'*4} PGD {'-'*4}")
    for eps in epsilons:
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            adv_images = pgd_attack(images, labels, model=model, epsilon=eps)

            with torch.no_grad():
                prediction = model.forward(adv_images).argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
            
        accuracy = 100 * (correct / total)
        pgd_accuracies.append(accuracy)
        print(f"PGD eps={eps:.3f} | Robust Acc: {accuracy:.2f}%")
            
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save({"epsilons": epsilons, "pgd_accuracies": pgd_accuracies}, os.path.join(RESULTS_DIR, "pgd_results.pth"))

    pass


def phase3_defense():
    """
    Phase 3 (Optional): Adversarial training + re-evaluate.
    Save checkpoint to checkpoints/adv_trained.pth
    """
    # TODO: Adversarial training
    model = BaselineCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "baseline.pth"), map_location=DEVICE
    ))

    optimizer = Adam(model.parameters(), lr=ADV_TRAIN_LEARNING_RATE)
    model, history = adversarial_training(
        model,
        train_loader,
        test_loader,
        scheduler=None,
        device=DEVICE,
        optimizer=optimizer,
        epsilon=ADV_TRAIN_EPSILON,
        alpha=PGD_ALPHA,
        pgd_steps=PGD_STEPS,
        num_epochs=ADV_TRAIN_EPOCHS,
    )
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "adv_trained.pth"))
    print("Phase 3: saved checkpoints/adv_trained.pth")

    # TODO: Evaluate robust model against FGSM and PGD
    criterion = torch.nn.CrossEntropyLoss()
    baseline_results = {
        "clean_accuracy": evaluate(model, test_loader, criterion, DEVICE),
        "fgsm_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "fgsm", FGSM_EPSILON
        ),
        "pgd_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "pgd", PGD_EPSILON,
            alpha=PGD_ALPHA, steps=PGD_STEPS
        ),
    }

    new_results = {
        "clean_accuracy": evaluate(model, test_loader, criterion, DEVICE),
        "fgsm_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "fgsm", FGSM_EPSILON
        ),
        "pgd_accuracy": evaluate_robustness(
            model, test_loader, DEVICE, "pgd", PGD_EPSILON,
            alpha=PGD_ALPHA, steps=PGD_STEPS
        ),
    }

    with open(os.path.join(RESULTS_DIR, "adv_trained_evaluation.json"), "w", encoding="utf-8") as output_file:
        json.dump({"baseline": baseline_results, "adversarially_trained": new_results}, output_file, indent=2)
    print("Phase 3 complete: saved adversarial-training evaluation results.")
    pass


def main():
    print("=" * 60)
    print("FGSM/PGD Adversarial Examples on CIFAR-10 Image Classifier")
    print("=" * 60)

    print("\n--- Phase 1: Training Baseline Model ---")
    # phase1_train_baseline()

    print("\n--- Phase 2: Attacking Baseline Model ---")
    # phase2_attack_baseline()

    print("\n--- Phase 3: Defense (Adversarial Training) ---")
    phase3_defense()

    print("\nDone! Check results/ for plots and analysis.")


if __name__ == "__main__":
    main()
