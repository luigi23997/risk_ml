"""
Linear model utilities.

Provides numerically stable implementations of linear regression
for use in time series models (e.g., AR) and future ML modules.

Design principles:
- Never use explicit matrix inversion
- Prefer least-squares or QR-based solutions
- Strict input validation
- Reusable across the entire library
"""

from __future__ import annotations

import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares (OLS) estimation using
    numerically stable least-squares solver.

    Solves:
        beta = argmin ||y - X beta||^2

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)

    Returns
    -------
    np.ndarray
        Estimated coefficients of shape (n_features,)
    """
    _validate_inputs(X, y)

    # Uses LAPACK-based least squares solver (stable)
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    if rank < X.shape[1]:
        raise np.linalg.LinAlgError(
            "Design matrix is rank deficient. "
            "OLS solution may not be unique."
        )

    return beta


def ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Ridge regression estimation.

    Solves:
        beta = (X'X + alpha I)^(-1) X'y

    but without explicit inversion.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    alpha : float
        Regularization strength (must be >= 0)

    Returns
    -------
    np.ndarray
        Estimated coefficients
    """
    _validate_inputs(X, y)

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    n_features = X.shape[1]

    XtX = X.T @ X
    regularization = alpha * np.eye(n_features)
    Xty = X.T @ y

    # Solve (XtX + alpha I) beta = Xty
    beta = np.linalg.solve(XtX + regularization, Xty)

    return beta


def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate linear model inputs.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of rows in X must match length of y.")

    if X.shape[0] == 0:
        raise ValueError("Empty input arrays are not allowed.")

    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input contains NaN values.")
