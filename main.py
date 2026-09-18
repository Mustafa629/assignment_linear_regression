"""
Main entry point: trains the final Linear Regression model (Batch GD,
lr=0.1, standardized features, 300 epochs -- the configuration selected
in Part 3a/3b/3c) on the California housing dataset, evaluates it on the
held-out test set, saves plots to figures/, and saves the trained model
to models/.

For the full experiment breakdown (learning-rate sweep, normalization
ablation, optimizer comparison, manual gradient-descent verification,
coefficient interpretation), see the scripts under experiments/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from Data_Handler import DataHandler
from Data_Preprocessor import DataPreprocessor
from LinearRegression_Model import LinearRegressionModel
from LinearRegression_Trainer import LinearRegressionTrainer
from Regression_Evaluator import RegressionEvaluator
from Regression_Visualizer import RegressionVisualizer
from Model_Persistence import ModelPersistence

ROOT = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(ROOT, "figures")
MODELS_DIR = os.path.join(ROOT, "models")

LEARNING_RATE = 0.1
N_EPOCHS = 300


class LinearRegression:

    def __init__(self):
        self.data_handler = DataHandler()
        self.preprocessor = DataPreprocessor()

        self.model = None
        self.trainer = None
        self.evaluator = RegressionEvaluator()
        self.visualizer = RegressionVisualizer()

    def run(self):

        # -------------------------------------------------
        # Load
        # -------------------------------------------------
        X, y = self.data_handler.load_data()

        # -------------------------------------------------
        # Split
        # -------------------------------------------------
        train_X, train_y, val_X, val_y, test_X, test_y = \
            self.data_handler.split_data(X, y)

        # -------------------------------------------------
        # Normalize (train-set statistics only, applied to val/test too)
        # -------------------------------------------------
        train_X = self.preprocessor.fit_transform(train_X)
        val_X = self.preprocessor.transform(val_X)
        test_X = self.preprocessor.transform(test_X)

        # -------------------------------------------------
        # Model
        # -------------------------------------------------
        self.model = LinearRegressionModel(
            n_features=train_X.shape[1]
        )

        # -------------------------------------------------
        # Trainer
        # -------------------------------------------------
        self.trainer = LinearRegressionTrainer(
            model=self.model,
            learning_rate=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            batch_size=32
        )

        # Students may switch between:
        #
        # train_batch_gd()
        # train_sgd()
        # train_minibatch_gd()
        train_losses, val_losses = self.trainer.train_batch_gd(
            train_X, train_y, val_X, val_y
        )
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss  : {val_losses[-1]:.4f}")

        # -------------------------------------------------
        # Testing
        # -------------------------------------------------
        test_y_hat = self.model.predict(test_X)
        results = self.evaluator.evaluate(test_y, test_y_hat)
        print("\nTest set performance:", results)

        print("\nLearned parameters:")
        for feature, weight in zip(
            self.data_handler.get_feature_names(), self.model.weights.flatten()
        ):
            print(f"  {feature}: {weight:+.4f}")
        print(f"  bias: {self.model.bias:+.4f}")

        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------
        os.makedirs(FIGURES_DIR, exist_ok=True)
        self.visualizer.plot_multiple_losses(
            {"train": train_losses, "val": val_losses},
            title="Batch GD - Train vs Validation Loss",
            save_path=os.path.join(FIGURES_DIR, "main_train_val_loss.png"),
        )
        self.visualizer.plot_predictions(
            test_y, test_y_hat,
            save_path=os.path.join(FIGURES_DIR, "main_actual_vs_predicted.png"),
        )

        # -------------------------------------------------
        # Save Model
        # -------------------------------------------------
        os.makedirs(MODELS_DIR, exist_ok=True)
        ModelPersistence.save(self.model, os.path.join(MODELS_DIR, "california_housing_model.pkl"))
        print(f"\nModel saved to {os.path.join(MODELS_DIR, 'california_housing_model.pkl')}")


if __name__ == "__main__":

    experiment = LinearRegression()
    experiment.run()
