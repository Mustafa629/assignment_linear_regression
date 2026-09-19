"""
Part 3 - Real image dataset (Kaggle frabbisw/facial-age), age regression.

Runs the exact same pipeline classes used for California housing and
COUGHVID (only DataHandler changes, per the assignment's OOP design)
against real downloaded face images. Grayscale, resized to 32x32 (1024
features/pixels), flattened -- a linear model over raw pixels.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Data_Handler_Image import ImageDataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer
from Regression_Evaluator import RegressionEvaluator
from Regression_Visualizer import RegressionVisualizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMAGE_SIZE = 32
N_EPOCHS = 1000
LEARNING_RATE = 0.0025
# NOTE: with 1024 raw-pixel features (many highly correlated/redundant --
# neighboring pixels), the loss surface is far more ill-conditioned than
# California housing's 8 largely-independent features. lr=0.05 (fine for
# housing) diverges here; lr=0.0025 was found stable via a manual sweep.


def main():
    dh = ImageDataHandler(
        root_dir=os.path.join(ROOT, "data", "face_age"),
        image_size=IMAGE_SIZE,
    )

    t0 = time.time()
    X, y = dh.load_data()
    print(f"Loaded {X.shape[0]} real face images, {X.shape[1]} features each, "
          f"in {time.time() - t0:.1f}s")
    print(f"Age label range: [{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")

    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)

    pre = DataPreprocessor()
    train_X = pre.fit_transform(train_X)
    val_X = pre.transform(val_X)
    test_X = pre.transform(test_X)

    model = LinearRegressionModel(n_features=train_X.shape[1])
    trainer = LinearRegressionTrainer(model=model, learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS)
    train_losses, val_losses = trainer.train_batch_gd(train_X, train_y, val_X, val_y)

    print(f"\nFinal train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss  : {val_losses[-1]:.4f}")

    evaluator = RegressionEvaluator()
    test_y_hat = model.predict(test_X)
    results = evaluator.evaluate(test_y, test_y_hat)
    print("Test set performance:", results)

    baseline_pred = train_y.mean()
    baseline_mse = evaluator.mse(test_y, baseline_pred)
    print(f"Baseline (predict training mean age) test MSE: {baseline_mse:.4f}")

    viz = RegressionVisualizer()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    viz.plot_multiple_losses(
        {"train": train_losses, "val": val_losses},
        title=f"Face-Age (real data, n={X.shape[0]}) - Train vs Val Loss",
        save_path=os.path.join(ROOT, "figures", "part3_image_real_loss.png"),
    )
    viz.plot_predictions(
        test_y, test_y_hat,
        save_path=os.path.join(ROOT, "figures", "part3_image_real_actual_vs_predicted.png"),
    )

    return results


if __name__ == "__main__":
    main()
