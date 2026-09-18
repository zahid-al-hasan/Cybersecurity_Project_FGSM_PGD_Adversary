"""
Adversarial Training Defense Implementation.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.

Adversarial training augments the training data with adversarial examples
generated during each epoch, forcing the model to learn robust features.
"""

import torch, tqdm
import torch.nn as nn
from attacks.pgd import pgd_attack
from utils.metrics import evaluate



def adversarial_training(model, train_loader, test_loader, optimizer, scheduler, device, epsilon, alpha, pgd_steps, num_epochs):
    """
    Train a model with PGD adversarial examples.

    For each batch:
        1. Generate PGD adversarial examples from the batch
        2. Compute loss on adversarial examples
        3. Update model weights

    Args:
        model: neural network to train
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        optimizer: optimizer (e.g., Adam)
        scheduler: learning rate scheduler
        device: "cuda" or "cpu"
        epsilon: PGD epsilon
        alpha: PGD step size
        pgd_steps: number of PGD iterations
        num_epochs: number of training epochs

    Returns:
        trained model, history dict with train/test losses and accuracies
    """
    
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
               "robust_acc": []}


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        batch_iterator = tqdm.tqdm(train_loader, desc=f"Adv epoch {epoch + 1}/{num_epochs}", leave=False)
        

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # TODO: Generate adversarial examples using PGD
            adv_images = pgd_attack(images, labels)

            # TODO: Forward pass on adversarial images
            outputs = model.forward(adv_images)
            loss = criterion(outputs, labels)

            # TODO: Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: Track loss and accuracy
            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            batch_iterator.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        
        # TODO: Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)


        # TODO: Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
        )

        if scheduler:
            scheduler.step()

    return model, history
