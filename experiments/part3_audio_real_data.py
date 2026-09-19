"""
Part 3 - Real audio dataset (COUGHVID), regression on 'cough_detected'.

Runs the exact same pipeline classes used for California housing (only
DataHandler changes, per the assignment's OOP design) against real
downloaded COUGHVID clips.

Each clip is decoded through a real ffmpeg subprocess (~0.1-0.15s/clip in
practice), so the full ~27,500-clip dataset takes roughly 45-60 minutes to
featurize. Set MAX_SAMPLES to an integer instead of None for a faster,
smaller-sample run.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Data_Handler_Audio import AudioDataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer
from Regression_Evaluator import RegressionEvaluator
from Regression_Visualizer import RegressionVisualizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAX_SAMPLES = None  # None = full ~27,500-clip dataset
N_EPOCHS = 200
LEARNING_RATE = 0.05


def main():
    dh = AudioDataHandler(
        audio_dir=os.path.join(ROOT, "data", "coughvid", "public_dataset"),
        csv_path=os.path.join(ROOT, "data", "coughvid", "public_dataset", "metadata_compiled.csv"),
        max_samples=MAX_SAMPLES,
    )

    t0 = time.time()
    X, y = dh.load_data()
    print(f"\nLoaded {X.shape[0]} real COUGHVID clips, {X.shape[1]} features each, "
          f"in {time.time() - t0:.1f}s")
    print(f"cough_detected label range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)

    pre = DataPreprocessor()
    train_X = pre.fit_transform(train_X)
    val_X = pre.transform(val_X)
    test_X = pre.transform(test_X)

    model = LinearRegressionModel(n_features=train_X.shape[1])
    trainer = LinearRegressionTrainer(model=model, learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS)
    train_losses, val_losses = trainer.train_batch_gd(train_X, train_y, val_X, val_y)

    print(f"\nFinal train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss  : {val_losses[-1]:.6f}")

    evaluator = RegressionEvaluator()
    test_y_hat = model.predict(test_X)
    results = evaluator.evaluate(test_y, test_y_hat)
    print("Test set performance:", results)

    # Baseline: MSE from just predicting the training mean for everyone,
    # to judge whether the model learned anything real about the audio.
    baseline_pred = train_y.mean()
    baseline_mse = evaluator.mse(test_y, baseline_pred)
    print(f"Baseline (predict training mean) test MSE: {baseline_mse:.6f}")

    print("\nLearned coefficients:")
    for feature, weight in zip(dh.get_feature_names(), model.weights.flatten()):
        print(f"  {feature:<20} {weight:+.4f}")
    print(f"  {'bias':<20} {model.bias:+.4f}")

    viz = RegressionVisualizer()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    viz.plot_multiple_losses(
        {"train": train_losses, "val": val_losses},
        title=f"COUGHVID (real data, n={X.shape[0]}) - Train vs Val Loss",
        save_path=os.path.join(ROOT, "figures", "part3_audio_real_loss.png"),
    )
    viz.plot_predictions(
        test_y, test_y_hat,
        save_path=os.path.join(ROOT, "figures", "part3_audio_real_actual_vs_predicted.png"),
    )

    return results


if __name__ == "__main__":
    main()
