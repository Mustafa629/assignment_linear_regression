"""
Part 3a - Learning-Rate Experiment.

Trains the SAME model (Batch GD) on the California housing dataset using
five different learning rates.

NOTE ON NORMALIZATION: an earlier run of this experiment on RAW features
diverged at EVERY one of the five required learning rates (even 0.00001),
because the "Population" feature has values up to ~35,000 while others are
single digits -- gradients tied to it are so large that no safe learning
rate exists at that scale. That finding is worth reporting for Part 3b.
To get a meaningful (non-degenerate) comparison across the five rates here,
this script standardizes features first, exactly as the main pipeline does.
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

LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01, 0.1]
N_EPOCHS = 200


def has_diverged(losses):
    return any(np.isnan(l) or np.isinf(l) or l > 1e6 for l in losses)


def has_converged(losses, rel_tol=1e-3, window=10):
    """Converged = loss barely changed (relatively) over the last `window` epochs."""
    if has_diverged(losses) or len(losses) < window + 1:
        return False
    recent = losses[-window:]
    relative_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-12)
    return relative_change < rel_tol


def main():
    dh = DataHandler()
    X, y = dh.load_data()
    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)

    pre = DataPreprocessor()
    train_X = pre.fit_transform(train_X)
    val_X = pre.transform(val_X)
    test_X = pre.transform(test_X)

    results = {}
    loss_curves = {}

    for lr in LEARNING_RATES:
        model = LinearRegressionModel(n_features=train_X.shape[1])
        trainer = LinearRegressionTrainer(model=model, learning_rate=lr, n_epochs=N_EPOCHS)
        train_losses, val_losses = trainer.train_batch_gd(train_X, train_y, val_X, val_y)

        diverged = has_diverged(train_losses)
        converged = has_converged(train_losses)

        final_train = train_losses[-1]
        final_val = val_losses[-1]

        results[lr] = {
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "diverged": diverged,
            "converged": converged,
        }
        loss_curves[f"lr={lr}"] = train_losses

        print(
            f"lr={lr:<10} final_train_loss={final_train:<15.6f} "
            f"final_val_loss={final_val:<15.6f} diverged={diverged} converged={converged}"
        )

    viz = RegressionVisualizer()
    viz.plot_multiple_losses(
        loss_curves,
        title="Part 3a: Loss vs Epoch (raw features, Batch GD)",
        save_path=os.path.join(os.path.dirname(__file__), "..", "figures", "part3a_loss_vs_epoch.png"),
    )

    return results


if __name__ == "__main__":
    main()
