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
├── data/                      # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/               # Saved model weights
└── results/                   # Plots and evaluation results
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train baseline model:
   ```
   python train.py
   ```

3. Evaluate attacks:
   ```
   python evaluate.py --model checkpoints/baseline.pth --attack fgsm --plot
   python evaluate.py --model checkpoints/baseline.pth --attack pgd --plot
   ```

4. Run full pipeline:
   ```
   python main.py
   ```

## References

- Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
