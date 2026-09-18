"""
Fast Gradient Sign Method (FGSM) Attack Implementation.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.

FGSM generates adversarial examples in a single step:
    x_adv = x + epsilon * sign(grad_x loss(model(x), y))

This is a fast but relatively weak attack.
"""

import torch
from models.cnn import BaselineCNN
from config import *
from torch.nn import CrossEntropyLoss


def fgsm_attack(images, labels, model=None, optimizer=None, epsilon=FGSM_EPSILON, criterion=None):
    """
    Generate adversarial examples using FGSM.

    Args:
        model: trained neural network
        images: batch of clean images (tensor)
        labels: true labels for the images (tensor)
        epsilon: perturbation magnitude (float)
        criterion: loss function (e.g., CrossEntropyLoss)

    Returns:
        adversarial images (tensor)
    """
    if model is None:
        model = BaselineCNN(num_classes=NUM_CLASSES)
    if criterion is None:
        criterion = CrossEntropyLoss()

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    images = images.detach().requires_grad_(True)
    was_training = model.training
    model.eval()

    try:
        output = model(images)
        loss = criterion(output, labels)
        gradient = torch.autograd.grad(loss, images)[0]
        adv_images = images + epsilon * gradient.sign()
        return torch.clamp(adv_images.detach(), 0, 1)
    finally:
        model.train(was_training)
