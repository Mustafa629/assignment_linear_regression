import numpy as np


class LinearRegressionModel:

    def __init__(self, n_features, seed=42):
        self.n_features = n_features
        self.seed = seed

        self.weights = None
        self.bias = None

        self.initialize_parameters()

    def initialize_parameters(self):
        self.weights = np.zeros((self.n_features, 1))
        self.bias = 0.0

    def predict(self, X):
        return X @ self.weights + self.bias

    def compute_loss(self, y, y_hat):
        error = y_hat - y
        return np.mean(error ** 2)

    def compute_gradient(self, X, y, y_hat):
        N = X.shape[0]
        error = y_hat - y                       # shape (N, 1)
        grad_w = (2 / N) * (X.T @ error)         # shape (n_features, 1)
        grad_b = (2 / N) * np.sum(error)         # scalar
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b, learning_rate):
        self.weights = self.weights - learning_rate * grad_w
        self.bias = self.bias - learning_rate * grad_b