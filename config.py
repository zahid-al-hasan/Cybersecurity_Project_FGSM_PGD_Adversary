"""
Configuration file for FGSM/PGD Adversarial Examples project.
Contains all hyperparameters and paths.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Dataset
DATASET = "CIFAR-10"
NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3

# Training
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# FGSM Attack
FGSM_EPSILON = 0.03  # epsilon for FGSM (CIFAR-10 pixel range is [0,1])

# PGD Attack
PGD_EPSILON = 0.03
PGD_ALPHA = 0.007    # step size
PGD_STEPS = 20       # number of iterations
PGD_RANDOM_START = True
PGD_NUM_RESTARTS = 5

# Adversarial Training
ADV_TRAIN_EPSILON = 0.03
ADV_TRAIN_EPOCHS = 50

# Device
DEVICE = "cuda"  # use "cuda" if GPU available, else "cpu"
