"""
Projected Gradient Descent (PGD) Attack Implementation.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.

PGD is the iterative, multi-step version of FGSM:
    x_(t+1) = clip(x_t + alpha * sign(grad_x loss(model(x_t), y)), x-epsilon, x+epsilon)

PGD is considered the strongest first-order adversary.
"""

import torch


def pgd_attack(model, images, labels, epsilon, alpha, steps, random_start=True):
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

    # Make a copy of images so original is not modified
    adv_images = images.clone().detach()

    # Optional: random start within epsilon ball
    if random_start:
        # TODO: Initialize adv_images with random perturbation
        pass

    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(steps):
        # TODO: Set requires_grad
        # TODO: Forward pass
        # TODO: Compute loss
        # TODO: Backward pass
        # TODO: Collect gradient
        # TODO: Update: adv_images = adv_images + alpha * sign(gradient)
        # TODO: Project back into epsilon ball: clip to [images-epsilon, images+epsilon]
        # TODO: Clip to valid pixel range [0, 1]
        pass

    return adv_images
