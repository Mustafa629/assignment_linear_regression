"""
Part 7 - Interpret the Learned Parameters.

Uses the same final configuration as Part 4 (Batch GD, lr=0.1, standardized
features, 300 epochs) so the coefficients discussed here match the model
actually evaluated on the test set.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from Data_Handler import DataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer

LEARNING_RATE = 0.1
N_EPOCHS = 300


def main():
    dh = DataHandler()
    X, y = dh.load_data()
    train_X, train_y, val_X, val_y, test_X, test_y = dh.split_data(X, y)
    feature_names = dh.get_feature_names()

    pre = DataPreprocessor()
    train_X_norm = pre.fit_transform(train_X)
    val_X_norm = pre.transform(val_X)

    model = LinearRegressionModel(n_features=train_X_norm.shape[1])
    trainer = LinearRegressionTrainer(model=model, learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS)
    trainer.train_batch_gd(train_X_norm, train_y, val_X_norm, val_y)

    weights = model.weights.flatten()

    print("Learned coefficients (on STANDARDIZED features -> comparable scale):")
    order = np.argsort(-np.abs(weights))
    for idx in order:
        print(f"  {feature_names[idx]:<12} {weights[idx]:+.4f}")
    print(f"  {'bias':<12} {model.bias:+.4f}")

    strongest_positive = feature_names[np.argmax(weights)]
    strongest_negative = feature_names[np.argmin(weights)]
    print(f"\nStrongest positive coefficient: {strongest_positive} ({weights.max():+.4f})")
    print(f"Strongest negative coefficient: {strongest_negative} ({weights.min():+.4f})")

    # ---------------------------------------------------------------
    # Correlation matrix among raw features -- concrete evidence for
    # "why coefficients should be interpreted carefully if features
    # are correlated."
    # ---------------------------------------------------------------
    print("\nFeature correlation matrix (raw features, training set):")
    corr = np.corrcoef(train_X.T)
    header = "".join(f"{n[:8]:>10}" for n in feature_names)
    print(" " * 12 + header)
    for i, name in enumerate(feature_names):
        row = "".join(f"{corr[i, j]:>10.2f}" for j in range(len(feature_names)))
        print(f"{name:<12}{row}")

    # Flag the most correlated OFF-DIAGONAL pair(s)
    corr_copy = corr.copy()
    np.fill_diagonal(corr_copy, 0)
    i, j = np.unravel_index(np.argmax(np.abs(corr_copy)), corr_copy.shape)
    print(f"\nMost correlated feature pair: {feature_names[i]} & {feature_names[j]} "
          f"(corr={corr[i, j]:.3f})")


if __name__ == "__main__":
    main()
