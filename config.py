"""
Configuration file for FGSM/PGD Adversarial Examples project.
Contains all hyperparameters and paths.
"""

import os
import random
import numpy as np
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Dataset
PARQUET_DATA_ROOT = os.path.join(BASE_DIR, "data", "cifar10")
DATASET = "CIFAR-10"
NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3

# Training
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
SEED = 406


def set_seed(seed=SEED):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

# FGSM Attack
FGSM_EPSILON = 0.03  # epsilon for FGSM (CIFAR-10 pixel range is [0,1])

# PGD Attack
PGD_EPSILON = 0.03
PGD_ALPHA = 0.007    # step size
PGD_STEPS = 2       # number of iterations
PGD_RANDOM_START = True
PGD_NUM_RESTARTS = 1

# Adversarial Training
ADV_TRAIN_EPSILON = 0.03
ADV_TRAIN_EPOCHS = 3

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
