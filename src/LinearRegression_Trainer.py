import numpy as np


class LinearRegressionTrainer:

    def __init__(
        self,
        model,
        learning_rate=0.001,
        n_epochs=100,
        batch_size=32,
        random_state=42
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.train_losses = []
        self.val_losses = []

    def train_batch_gd(self, train_X, train_y, val_X, val_y):
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.n_epochs):
            y_hat = self.model.predict(train_X)
            train_loss = self.model.compute_loss(train_y, y_hat)

            grad_w, grad_b = self.model.compute_gradient(train_X, train_y, y_hat)
            self.model.update_parameters(grad_w, grad_b, self.learning_rate)

            val_loss = self.validate(val_X, val_y)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        return self.train_losses, self.val_losses

    def train_sgd(self, train_X, train_y, val_X, val_y):
        self.train_losses = []
        self.val_losses = []
        rng = np.random.default_rng(self.random_state)
        n_samples = train_X.shape[0]

        for epoch in range(self.n_epochs):
            shuffled_indices = rng.permutation(n_samples)
            X_shuffled = train_X[shuffled_indices]
            y_shuffled = train_y[shuffled_indices]

            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]   # keep 2D shape (1, n_features)
                y_i = y_shuffled[i:i + 1]   # keep 2D shape (1, 1)

                y_hat = self.model.predict(x_i)
                grad_w, grad_b = self.model.compute_gradient(x_i, y_i, y_hat)
                self.model.update_parameters(grad_w, grad_b, self.learning_rate)

            # Measure loss on the FULL sets once per epoch (not the last sample only).
            train_loss = self.validate(train_X, train_y)
            val_loss = self.validate(val_X, val_y)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        return self.train_losses, self.val_losses

    def train_minibatch_gd(self, train_X, train_y, val_X, val_y):
        self.train_losses = []
        self.val_losses = []
        rng = np.random.default_rng(self.random_state)
        n_samples = train_X.shape[0]

        for epoch in range(self.n_epochs):
            shuffled_indices = rng.permutation(n_samples)
            X_shuffled = train_X[shuffled_indices]
            y_shuffled = train_y[shuffled_indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_hat = self.model.predict(X_batch)
                grad_w, grad_b = self.model.compute_gradient(X_batch, y_batch, y_hat)
                self.model.update_parameters(grad_w, grad_b, self.learning_rate)

            train_loss = self.validate(train_X, train_y)
            val_loss = self.validate(val_X, val_y)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        return self.train_losses, self.val_losses

    def validate(self, X, y):
        y_hat = self.model.predict(X)
        return self.model.compute_loss(y, y_hat)