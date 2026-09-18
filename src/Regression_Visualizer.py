import numpy as np
import matplotlib.pyplot as plt


class RegressionVisualizer:

    @staticmethod
    def plot_loss(losses, title="Training Loss", save_path=None):
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    @staticmethod
    def plot_multiple_losses(loss_dictionary, title="Loss Comparison", save_path=None):
        plt.figure()
        for label, losses in loss_dictionary.items():
            plt.plot(losses, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(title)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()

    @staticmethod
    def plot_predictions(y, y_hat, save_path=None):
        y = y.flatten()
        y_hat = y_hat.flatten()

        plt.figure()
        plt.scatter(y, y_hat, alpha=0.4, s=10)

        min_val = min(y.min(), y_hat.min())
        max_val = max(y.max(), y_hat.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red")  # y = x reference line

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs. Predicted")
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()