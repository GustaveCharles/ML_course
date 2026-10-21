"""Project 1 methods, adapted from the course Lab 2, 4, and 5 corrections.

The public interface uses one-dimensional target and weight arrays. Each loss is
the unpenalized, mean per-example loss evaluated at the returned weights.
"""

import numpy as np


def _mse(y, tx, w):
    """Lab 2/4 convention: half the mean squared residual."""
    return 0.5 * np.mean((y - tx @ w) ** 2)


def _sigmoid(z):
    """The Lab 5 sigmoid, with clipping to avoid overflow for large inputs."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -709.0, 709.0)))


def _logistic_loss(y, tx, w):
    """Lab 5 negative log likelihood in a numerically stable form."""
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit least squares by full-batch gradient descent (Lab 2)."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        error = y - tx @ w
        gradient = -(tx.T @ error) / len(y)
        w -= gamma * gradient
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit least squares with one random example per update (Lab 2)."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        index = np.random.randint(len(y))
        error = y[index] - tx[index] @ w
        w += gamma * tx[index] * error
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Solve the unregularized normal equations (Lab 4)."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Solve the Lab 4 ridge normal equations; return unpenalized MSE."""
    n, d = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit binary logistic regression by full-batch descent (Lab 5)."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / len(y)
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit Lab 5 L2 logistic regression; omit penalty from returned loss."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)
