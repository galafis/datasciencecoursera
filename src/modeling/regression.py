"""
Regression Module

Provides implementations of linear regression and logistic regression
using numpy and scipy, without relying on scikit-learn.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class LinearRegression:
    """
    Ordinary Least Squares (OLS) linear regression.

    Supports single and multiple features using the normal equation.
    """

    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the model using the normal equation: beta = (X^T X)^-1 X^T y.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or (n_samples,).
            y: Target vector of shape (n_samples,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        X_b = np.column_stack([np.ones(n_samples), X])

        beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.intercept = float(beta[0])
        self.coefficients = beta[1:]
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or (n_samples,).

        Returns:
            Predicted values as 1D array.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coefficients + self.intercept

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R-squared (coefficient of determination).

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            R-squared value.
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def get_params(self) -> Dict[str, object]:
        """Return model parameters."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
        }


class LogisticRegression:
    """
    Binary logistic regression using gradient descent.

    Uses the sigmoid function and binary cross-entropy loss.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self._fitted = False
        self.loss_history: list = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model using gradient descent.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or (n_samples,).
            y: Binary target vector of shape (n_samples,) with values 0 or 1.

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)

            eps = 1e-15
            loss = -np.mean(y * np.log(predictions + eps) + (1 - y) * np.log(1 - predictions + eps))
            self.loss_history.append(loss)

            error = predictions - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if len(self.loss_history) > 1 and abs(self.loss_history[-2] - loss) < self.tolerance:
                break

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1.

        Args:
            X: Feature matrix.

        Returns:
            Probabilities as 1D array.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X: Feature matrix.
            threshold: Decision threshold (default 0.5).

        Returns:
            Binary predictions as 1D array.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Feature matrix.
            y: True binary labels.

        Returns:
            Accuracy as float between 0 and 1.
        """
        y = np.asarray(y, dtype=float)
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def get_params(self) -> Dict[str, object]:
        """Return model parameters."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_iterations_run": len(self.loss_history),
        }
