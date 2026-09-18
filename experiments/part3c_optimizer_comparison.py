"""
Part 3c - Batch GD vs SGD vs Mini-Batch GD.

Each method gets its OWN stable learning rate rather than one shared rate,
because they update parameters at very different frequencies per epoch:

  Batch GD      : 1 update/epoch      -> stable up to lr ~ 0.1   (see Part 3a)
  Mini-batch GD : ~452 updates/epoch  -> only stable up to lr ~ 0.001
  SGD           : 14,448 updates/epoch-> only stable up to lr ~ 0.0001

(All three were verified experimentally: the "natural" shared choice of
lr=0.01 that batch GD handles fine makes both SGD and mini-batch diverge.)
This mirrors a real lesson: more frequent updates per epoch compounds
small per-step errors faster, so the safe learning rate shrinks as update
frequency increases.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from Data_Handler import DataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer
from Regression_Evaluator import RegressionEvaluator
from Regression_Visualizer import RegressionVisualizer

N_EPOCHS = 50
BATCH_SIZE = 32

CONFIGS = {
    "Batch GD": {"lr": 0.1, "method": "train_batch_gd"},
    "SGD": {"lr": 0.0001, "method": "train_sgd"},
    "Mini-Batch GD": {"lr": 0.001, "method": "train_minibatch_gd"},
}


def noisiness(losses):
    """Fraction of epoch-to-epoch steps where the loss went UP instead of down --
    a direct, objective sign of noisy updates (a perfectly monotonic curve scores 0)."""
    diffs = np.diff(losses)
    return float(np.mean(diffs > 0))


def main():
    dh = DataHandler()
    X, y = dh.load_data()
    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)

    pre = DataPreprocessor()
    train_X = pre.fit_transform(train_X)
    val_X = pre.transform(val_X)
    test_X = pre.transform(test_X)

    evaluator = RegressionEvaluator()
    loss_curves = {}
    summary = []

    for name, cfg in CONFIGS.items():
        model = LinearRegressionModel(n_features=train_X.shape[1])
        trainer = LinearRegressionTrainer(
            model=model, learning_rate=cfg["lr"], n_epochs=N_EPOCHS, batch_size=BATCH_SIZE
        )
        train_fn = getattr(trainer, cfg["method"])

        t0 = time.perf_counter()
        train_losses, val_losses = train_fn(train_X, train_y, val_X, val_y)
        elapsed = time.perf_counter() - t0

        test_y_hat = model.predict(test_X)
        test_mse = evaluator.mse(test_y, test_y_hat)

        loss_curves[name] = train_losses
        summary.append({
            "name": name,
            "lr": cfg["lr"],
            "time_sec": elapsed,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "test_mse": test_mse,
            "noisiness": noisiness(train_losses),
        })

        print(f"{name:<15} lr={cfg['lr']:<8} time={elapsed:6.3f}s  "
              f"final_train_loss={train_losses[-1]:.4f}  test_MSE={test_mse:.4f}  "
              f"noisiness={noisiness(train_losses):.5f}")

    viz = RegressionVisualizer()
    viz.plot_multiple_losses(
        loss_curves,
        title="Part 3c: Batch GD vs SGD vs Mini-Batch GD",
        save_path=os.path.join(os.path.dirname(__file__), "..", "figures", "part3c_optimizer_comparison.png"),
    )

    return summary


if __name__ == "__main__":
    main()
