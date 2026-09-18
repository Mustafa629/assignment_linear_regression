"""
Part 3b - Normalization Experiment.

Experiment A: train on raw (unnormalized) features.
Experiment B: train on standardized features (train-set mean/std only,
              applied to val/test too -- see Data_Preprocessor).

Two comparisons are run:
  1. Head-to-head at the SAME fixed learning rate, to compare convergence
     speed / stability / final losses directly.
  2. A learning-rate sensitivity sweep for each condition, to see how much
     the *range* of usable learning rates differs between raw and
     standardized features.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from Data_Handler import DataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer
from Regression_Visualizer import RegressionVisualizer

FIXED_LR = 0.01
N_EPOCHS = 200

SENSITIVITY_LRS_RAW = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
SENSITIVITY_LRS_NORM = [0.00001, 0.0001, 0.001, 0.01, 0.1]


def has_diverged(losses):
    return any(np.isnan(l) or np.isinf(l) or l > 1e6 for l in losses)


def run(train_X, train_y, val_X, val_y, lr, n_epochs=N_EPOCHS):
    model = LinearRegressionModel(n_features=train_X.shape[1])
    trainer = LinearRegressionTrainer(model=model, learning_rate=lr, n_epochs=n_epochs)
    train_losses, val_losses = trainer.train_batch_gd(train_X, train_y, val_X, val_y)
    return train_losses, val_losses


def main():
    dh = DataHandler()
    X, y = dh.load_data()
    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)

    pre = DataPreprocessor()
    train_X_norm = pre.fit_transform(train_X)
    val_X_norm = pre.transform(val_X)

    # -----------------------------------------------------------------
    # 1. Head-to-head at the same learning rate
    # -----------------------------------------------------------------
    print("=" * 70)
    print(f"Head-to-head at the SAME learning rate = {FIXED_LR}")
    print("=" * 70)

    tl_a, vl_a = run(train_X, train_y, val_X, val_y, FIXED_LR)
    tl_b, vl_b = run(train_X_norm, train_y, val_X_norm, val_y, FIXED_LR)

    diverged_a = has_diverged(tl_a)
    diverged_b = has_diverged(tl_b)

    print(f"Experiment A (raw)        : diverged={diverged_a}  "
          f"final_train={'nan/inf' if diverged_a else f'{tl_a[-1]:.6f}'}")
    print(f"Experiment B (standardized): diverged={diverged_b}  "
          f"final_train={'nan/inf' if diverged_b else f'{tl_b[-1]:.6f}'}  "
          f"final_val={'nan/inf' if diverged_b else f'{vl_b[-1]:.6f}'}")

    # Plot only the non-diverging curve(s) sensibly; still show both, clipped view helps.
    viz = RegressionVisualizer()
    viz.plot_multiple_losses(
        {"A: raw features": tl_a, "B: standardized features": tl_b},
        title=f"Part 3b: Raw vs Standardized (lr={FIXED_LR})",
        save_path=os.path.join(os.path.dirname(__file__), "..", "figures", "part3b_normalization_comparison.png"),
    )

    # Experiment A's curve shoots to inf/NaN, which flattens B's curve to invisible
    # on a shared axis. A second, zoomed-in plot of B alone makes its shape visible.
    viz.plot_loss(
        tl_b,
        title=f"Part 3b: Standardized features only (lr={FIXED_LR}, zoomed in)",
        save_path=os.path.join(os.path.dirname(__file__), "..", "figures", "part3b_normalization_zoomed.png"),
    )

    # -----------------------------------------------------------------
    # 2. Learning-rate sensitivity
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("Learning-rate sensitivity")
    print("=" * 70)

    print("\nExperiment A (raw features):")
    stable_a = 0
    for lr in SENSITIVITY_LRS_RAW:
        tl, vl = run(train_X, train_y, val_X, val_y, lr)
        d = has_diverged(tl)
        stable_a += int(not d)
        print(f"  lr={lr:<10} diverged={d}  final_train_loss={'nan/inf' if d else f'{tl[-1]:.4f}'}")

    print("\nExperiment B (standardized features):")
    stable_b = 0
    for lr in SENSITIVITY_LRS_NORM:
        tl, vl = run(train_X_norm, train_y, val_X_norm, val_y, lr)
        d = has_diverged(tl)
        stable_b += int(not d)
        print(f"  lr={lr:<10} diverged={d}  final_train_loss={'nan/inf' if d else f'{tl[-1]:.4f}'}")

    print(f"\nStable (non-diverging) rates: raw = {stable_a}/{len(SENSITIVITY_LRS_RAW)}, "
          f"standardized = {stable_b}/{len(SENSITIVITY_LRS_NORM)}")


if __name__ == "__main__":
    main()
