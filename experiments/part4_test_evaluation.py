"""
Part 4 - Prediction and Evaluation on the unseen test set.

Uses the best configuration found in Part 3: Batch GD, lr=0.1, standardized
features (train-set mean/std applied to val/test). Trained long enough to
reach the plateau observed in Part 3a (loss stopped improving meaningfully
past ~150-200 epochs).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from Data_Handler import DataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer
from Regression_Evaluator import RegressionEvaluator
from Regression_Visualizer import RegressionVisualizer

LEARNING_RATE = 0.1
N_EPOCHS = 300


def main():
    dh = DataHandler()
    X, y = dh.load_data()
    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)

    pre = DataPreprocessor()
    train_X = pre.fit_transform(train_X)
    val_X = pre.transform(val_X)
    test_X = pre.transform(test_X)

    model = LinearRegressionModel(n_features=train_X.shape[1])
    trainer = LinearRegressionTrainer(model=model, learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS)
    train_losses, val_losses = trainer.train_batch_gd(train_X, train_y, val_X, val_y)

    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss  : {val_losses[-1]:.4f}")

    # -----------------------------------------------------------------
    # Evaluate on the UNSEEN test set -- the model never saw this data,
    # not for training, not for the learning-rate/normalization choices.
    # -----------------------------------------------------------------
    test_y_hat = model.predict(test_X)

    evaluator = RegressionEvaluator()
    results = evaluator.evaluate(test_y, test_y_hat)
    print("\nTest-set performance:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # For context: also report train/val performance with the same metrics,
    # so the report can explicitly show whether the model over/underfit.
    train_y_hat = model.predict(train_X)
    val_y_hat = model.predict(val_X)
    print("\nFor comparison -- train-set MSE:", evaluator.mse(train_y, train_y_hat))
    print("For comparison -- val-set MSE  :", evaluator.mse(val_y, val_y_hat))

    # Largest individual test-set errors -- concretely answers "what does a
    # point far from the diagonal represent?"
    abs_errors = np.abs(test_y_hat - test_y).flatten()
    worst_idx = np.argsort(abs_errors)[-5:][::-1]
    print("\n5 largest test-set errors (actual vs predicted, in $100k units):")
    for idx in worst_idx:
        print(f"  actual={test_y[idx, 0]:.3f}  predicted={test_y_hat[idx, 0]:.3f}  "
              f"abs_error={abs_errors[idx]:.3f}")

    viz = RegressionVisualizer()
    viz.plot_predictions(
        test_y, test_y_hat,
        save_path=os.path.join(os.path.dirname(__file__), "..", "figures", "part4_actual_vs_predicted.png"),
    )

    return results


if __name__ == "__main__":
    main()
