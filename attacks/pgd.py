"""
Projected Gradient Descent (PGD) Attack Implementation.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.

PGD is the iterative, multi-step version of FGSM:
    x_(t+1) = clip(x_t + alpha * sign(grad_x loss(model(x_t), y)), x-epsilon, x+epsilon)

PGD is considered the strongest first-order adversary.
"""

import torch
from config import *
from models.cnn import BaselineCNN
from torch.nn import CrossEntropyLoss


def pgd_attack(images, labels, model=None, optimizer=None, criterion=None,
               epsilon=PGD_EPSILON, alpha=PGD_ALPHA, steps=PGD_STEPS,
               random_start=PGD_RANDOM_START, num_restarts=PGD_NUM_RESTARTS):
    """
    Generate adversarial examples using PGD.

    Args:
        model: trained neural network
        images: batch of clean images (tensor)
        labels: true labels for the images (tensor)
        epsilon: maximum perturbation (float)
        alpha: step size per iteration (float)
        steps: number of PGD iterations (int)
        random_start: whether to start from random point in epsilon ball (bool)
        num_restarts: number of attack initializations

    Returns:
        adversarial images (tensor)
    """

    if model is None:
        model = BaselineCNN(num_classes=NUM_CLASSES)
    if criterion is None:
        criterion = CrossEntropyLoss()
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least 1")

    clean_images = images.detach()
    was_training = model.training
    model.eval()
    best_images = None
    best_loss = None

    try:
        for _ in range(num_restarts):
            if random_start:
                perturbation = torch.empty_like(clean_images).uniform_(-epsilon, epsilon)
                adv_images = torch.clamp(clean_images + perturbation, 0, 1)
            else:
                adv_images = clean_images.clone()

            for _ in range(steps):
                adv_images = adv_images.detach().requires_grad_(True)
                output = model(adv_images)
                loss = criterion(output, labels)
                gradient = torch.autograd.grad(loss, adv_images)[0]

                with torch.no_grad():
                    adv_images = adv_images + alpha * gradient.sign()
                    lower_bound = clean_images - epsilon
                    upper_bound = clean_images + epsilon
                    adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)
                    adv_images = torch.clamp(adv_images, 0, 1)

            with torch.no_grad():
                candidate_loss = criterion(model(adv_images), labels).detach()
            if best_loss is None or candidate_loss > best_loss:
                best_loss = candidate_loss
                best_images = adv_images.detach().clone()

        return best_images
    finally:
        model.train(was_training)
