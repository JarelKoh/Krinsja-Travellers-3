# Krinsja-Travellers-3

This repository contains the code, data, and report for **FYS-STK3155 Project 3**, which investigates deep learning–based approaches for breast cancer image segmentation using convolutional neural networks (CNNs) and U-Net architectures.

The project includes:
- Reproducible Python code converted from Jupyter notebooks
- A locally stored dataset
- The final report written in Overleaf (PDF)

---

## Repository Structure

```text
Krinsja-Travellers-3/
├── Code/
│   ├── main.py                 # Main training and evaluation script
│   ├── hyperparameter.py       # Bias–variance and hyperparameter analysis
│   ├── models.py               # CNN and U-Net model definitions
│   ├── losses.py               # BCE, weighted BCE, Dice, focal, combined losses
│   ├── data_utils.py           # Data loading and preprocessing utilities
│   ├── metrics.py              # Dice score and evaluation helpers
│   └── requirements.txt
│
├── Report/
│   └── FYS_STK3155_Project3.pdf # Final report (Overleaf export)
│
└── README.md
