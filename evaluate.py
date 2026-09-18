"""
Evaluation script for testing attacks and defenses.

Usage:
    python evaluate.py --model checkpoints/baseline.pth --attack fgsm
    python evaluate.py --model checkpoints/baseline.pth --attack pgd
    python evaluate.py --model checkpoints/adv_trained.pth --attack pgd
"""

import argparse
import json
import os
import torch

from models.cnn import BaselineCNN
from dataset_loader import get_dataloaders
from utils.metrics import evaluate_model, evaluate_robustness, evaluate_over_epsilons
from utils.visualization import plot_accuracy_vs_epsilon, plot_examples
from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate attacks and defenses")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "fgsm", "pgd"], help="Attack type")
    parser.add_argument("--epsilon", type=float, default=FGSM_EPSILON, help="Epsilon")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate both attacks over the configured epsilon values",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "evaluation.json"),
        help="Path for the JSON results file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"model checkpoint not found: {args.model}")

    device = torch.device(DEVICE)
    _, test_loader = get_dataloaders(PARQUET_DATA_ROOT, args.batch_size)
    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    results = {
        "model": os.path.abspath(args.model),
        "device": str(device),
        "clean_accuracy": evaluate_model(model, test_loader, device),
    }

    if args.attack != "none":
        results["attack"] = args.attack
        results["epsilon"] = args.epsilon
        results["robust_accuracy"] = evaluate_robustness(
            model, test_loader, device, args.attack, args.epsilon
        )

    if args.sweep or args.plot:
        epsilons = [0.0, 0.01, 0.02, 0.03]
        results["epsilons"] = epsilons
        results["fgsm_accuracies"] = evaluate_over_epsilons(
            model, test_loader, device, "fgsm", epsilons
        )
        results["pgd_accuracies"] = evaluate_over_epsilons(
            model, test_loader, device, "pgd", epsilons
        )
        if args.plot:
            plot_accuracy_vs_epsilon(
                epsilons,
                results["fgsm_accuracies"],
                results["pgd_accuracies"],
            )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
