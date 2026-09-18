import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


class DataHandler:

    def __init__(self, test_size=0.15, val_size=0.15, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.feature_names = None

    def load_data(self):
        dataset = fetch_california_housing()

        X = dataset.data
        y = dataset.target.reshape(-1, 1)

        self.feature_names = dataset.feature_names

        return X, y

    def split_data(self, X, y):
        # Step 1: peel off the test set from the full dataset.
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Step 2: split what's left into train/val. Rescale val_size so it's
        # still val_size of the ORIGINAL dataset, not of the remainder.
        val_ratio_of_remainder = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio_of_remainder,
            random_state=self.random_state
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_feature_names(self):
        return self.feature_names