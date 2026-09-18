"""
Fast Gradient Sign Method (FGSM) Attack Implementation.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.

FGSM generates adversarial examples in a single step:
    x_adv = x + epsilon * sign(grad_x loss(model(x), y))

This is a fast but relatively weak attack.
"""

import torch
from models.cnn import BaselineCNN
from torch.optim import Adam
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
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    if criterion is None:
        criterion = CrossEntropyLoss()


    # Set images to require gradient
    images = images.detach().requires_grad_(True)

    # TODO: Forward pass - get model predictions
    output = model.forward(images)

    # TODO: Compute loss
    loss = criterion(output, labels)

    # TODO: Backward pass - compute gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO: Collect gradient of loss w.r.t. input images
    gradient = images.grad

    # TODO: Create adversarial example: x_adv = x + epsilon * sign(gradient)
    grad_sign = gradient/abs(gradient) if gradient != 0 else 0
    adv_images = images + epsilon * grad_sign

    # TODO: Clip adversarial images to valid range [0, 1]
    adv_images = torch.clamp(adv_images, 0, 1)

    pass
