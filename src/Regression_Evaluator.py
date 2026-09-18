import numpy as np


class RegressionEvaluator:

    @staticmethod
    def mse(y, y_hat):
        return np.mean((y_hat - y) ** 2)

    @staticmethod
    def rmse(y, y_hat):
        return np.sqrt(np.mean((y_hat - y) ** 2))

    @staticmethod
    def mae(y, y_hat):
        return np.mean(np.abs(y_hat - y))

    def evaluate(self, y, y_hat):
        return {
            "MSE": self.mse(y, y_hat),
            "RMSE": self.rmse(y, y_hat),
            "MAE": self.mae(y, y_hat),
        }
