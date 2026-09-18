# FGSM/PGD Adversarial Examples on Image Classifier

Cybersecurity Sessional Project

## Project Structure

```
FGSM_PGD_Adversary/
├── config.py                  # Hyperparameters and paths
├── train.py                   # Baseline model training
├── evaluate.py                # Attack evaluation script
├── main.py                    # Full pipeline entry point
├── requirements.txt           # Dependencies
├── models/
│   ├── __init__.py
│   └── cnn.py                 # Baseline CNN architecture
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py                # FGSM attack
│   └── pgd.py                 # PGD attack
├── defenses/
│   ├── __init__.py
│   └── adversarial_training.py # Adversarial training defense
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # Plotting utilities
│   └── metrics.py             # Evaluation metrics
├── data/                      # Local CIFAR-10 Parquet dataset
├── checkpoints/               # Saved model weights
└── results/                   # Plots and evaluation results
```

## How to Run

1. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

2. Place the Parquet files at:
   ```
   data/cifar10/plain_text/train-00000-of-00001.parquet
   data/cifar10/plain_text/test-00000-of-00001.parquet
   ```
   The loader expects CIFAR-10 RGB images in the `img` column and integer labels
   from `0` to `9` in the `label` column. `ToTensor()` converts images to
   `[3, 32, 32]` tensors in the `[0, 1]` range; no mean/std normalization is used.

3. Train baseline model:
   ```
   python train.py
   ```

4. Evaluate attacks:
   ```
   python evaluate.py --model checkpoints/baseline.pth --attack fgsm --plot
   python evaluate.py --model checkpoints/baseline.pth --attack pgd --plot
   ```

5. Run full pipeline:
   ```
   python main.py
   ```

The full pipeline writes checkpoints to `checkpoints/` and plots/JSON metrics
to `results/`. The baseline uses 20 training epochs. The adversarially trained
model uses 50 epochs with PGD, epsilon `0.03`, step size `0.007`, and 20 steps.
PGD evaluation uses five random restarts. Runtime depends heavily on whether
CUDA is available.

## Threat Model

FGSM and PGD are untargeted white-box attacks constrained by an `L_inf` budget.
The default epsilon values are expressed in the loader's `[0, 1]` tensor scale.
Adversarial training uses PGD-generated examples and is evaluated with both
FGSM and PGD.

## References

- Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
