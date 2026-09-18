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
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


def pgd_attack(images, labels, model=None, optimizer=None, criterion=None, epsilon=PGD_EPSILON, alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START):
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

    Returns:
        adversarial images (tensor)
    """

    if model is None:
        model = BaselineCNN(num_classes=NUM_CLASSES)
    if optimizer is None:
        optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
    if criterion is None:
        criterion = CrossEntropyLoss()

    # Make a copy of images so original is not modified
    adv_images = images.clone().detach()

    # Optional: random start within epsilon ball
    if random_start:
        # TODO: Initialize adv_images with random perturbation

        pass

    # criterion = torch.nn.CrossEntropyLoss()

    for _ in range(steps):
        # TODO: Set requires_grad
        adv_images = adv_images.requires_grad_(True)

        # TODO: Forward pass
        output = model.forward(adv_images)

        # TODO: Compute loss
        loss = criterion(output, labels)

        # TODO: Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: Collect gradient
        gradient = adv_images.grad

        # TODO: Update: adv_images = adv_images + alpha * sign(gradient)
        grad_sign = gradient/abs(gradient) if gradient != 0 else 0
        adv_images += alpha * grad_sign

        # TODO: Project back into epsilon ball: clip to [images-epsilon, images+epsilon]
        adv_images = torch.clamp(adv_images, images - epsilon, images + epsilon)

        # TODO: Clip to valid pixel range [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1)
        pass

    return adv_images
