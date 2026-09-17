"""
Fast Gradient Sign Method (FGSM) Attack Implementation.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.

FGSM generates adversarial examples in a single step:
    x_adv = x + epsilon * sign(grad_x loss(model(x), y))

This is a fast but relatively weak attack.
"""

import torch


def fgsm_attack(model, images, labels, epsilon, criterion):
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

    # Set images to require gradient
    images = images.detach().requires_grad_(True)

    # TODO: Forward pass - get model predictions
    # TODO: Compute loss
    # TODO: Backward pass - compute gradients
    # TODO: Collect gradient of loss w.r.t. input images
    # TODO: Create adversarial example: x_adv = x + epsilon * sign(gradient)
    # TODO: Clip adversarial images to valid range [0, 1]

    pass
