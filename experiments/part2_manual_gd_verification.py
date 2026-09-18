"""
Part 2 - Manual Gradient-Descent Calculation, verified against LinearRegressionModel.

This script feeds the exact numbers from the assignment's manual-calculation
problem into the real, trained-from-scratch LinearRegressionModel class, so
the hand-computed values (done on paper) can be checked against code output
to numerical precision.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from LinearRegression_Model import LinearRegressionModel

# ---------------------------------------------------------------------------
# Given in the assignment
# ---------------------------------------------------------------------------
X = np.array([[1.0], [2.0], [3.0]])   # single feature, 3 samples -> shape (3, 1)
Y = np.array([[3.0], [5.0], [7.0]])   # shape (3, 1)

theta_1_init = 0.5
theta_0_init = 0.7
learning_rate = 0.01

# ---------------------------------------------------------------------------
# Load the given starting parameters into the model
# ---------------------------------------------------------------------------
model = LinearRegressionModel(n_features=1)
model.weights = np.array([[theta_1_init]])
model.bias = theta_0_init

# ---------------------------------------------------------------------------
# Step 1-2: forward pass + per-sample error
# ---------------------------------------------------------------------------
y_hat = model.predict(X)
errors = y_hat - Y

print("y_hat        :", y_hat.flatten())
print("errors (yhat-y):", errors.flatten())

# ---------------------------------------------------------------------------
# Step 3: MSE
# ---------------------------------------------------------------------------
mse = model.compute_loss(Y, y_hat)
print("MSE           :", mse)

# ---------------------------------------------------------------------------
# Step 4-5: gradients
# ---------------------------------------------------------------------------
grad_w, grad_b = model.compute_gradient(X, Y, y_hat)
print("dL/d_theta1   :", grad_w.flatten())
print("dL/d_theta0   :", grad_b)

# ---------------------------------------------------------------------------
# Step 6: parameter update
# ---------------------------------------------------------------------------
model.update_parameters(grad_w, grad_b, learning_rate)
print("theta1_new    :", model.weights.flatten())
print("theta0_new    :", model.bias)

# ---------------------------------------------------------------------------
# Compare against hand-calculated expected values
# ---------------------------------------------------------------------------
expected_y_hat = np.array([1.2, 1.7, 2.2])
expected_errors = np.array([-1.8, -3.3, -4.8])
expected_mse = 12.39
expected_grad_w = -15.2
expected_grad_b = -6.6
expected_theta1 = 0.652
expected_theta0 = 0.766

assert np.allclose(y_hat.flatten(), expected_y_hat), "y_hat mismatch"
assert np.allclose(errors.flatten(), expected_errors), "errors mismatch"
assert np.isclose(mse, expected_mse), "MSE mismatch"
assert np.isclose(grad_w.flatten()[0], expected_grad_w), "grad_w mismatch"
assert np.isclose(grad_b, expected_grad_b), "grad_b mismatch"
assert np.isclose(model.weights.flatten()[0], expected_theta1), "theta1 update mismatch"
assert np.isclose(model.bias, expected_theta0), "theta0 update mismatch"

print("\nAll values match the manual calculation. Verification passed.")
