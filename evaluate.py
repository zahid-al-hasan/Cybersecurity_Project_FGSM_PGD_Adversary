"""
Evaluation script for testing attacks and defenses.

Usage:
    python evaluate.py --model checkpoints/baseline.pth --attack fgsm
    python evaluate.py --model checkpoints/baseline.pth --attack pgd
    python evaluate.py --model checkpoints/adv_trained.pth --attack pgd
"""

import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import BaselineCNN
from utils.metrics import evaluate_model, evaluate_robustness, evaluate_over_epsilons
from utils.visualization import plot_accuracy_vs_epsilon, plot_examples
from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate attacks and defenses")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "fgsm", "pgd"], help="Attack type")
    parser.add_argument("--epsilon", type=float, default=FGSM_EPSILON, help="Epsilon")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: Set device
    # TODO: Load CIFAR-10 test set
    # TODO: Load model from checkpoint
    # TODO: Evaluate clean accuracy
    # TODO: If attack specified, evaluate robust accuracy
    # TODO: If plotting, generate accuracy vs epsilon curves
    # TODO: Save results
    pass


if __name__ == "__main__":
    main()
