"""
Likelihood functions for time series and risk models.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

def gaussian_loglik(residuals: np.ndarray) -> float:
    """
    Compute Gaussian log-likelihood for residuals.

    Assumes residuals ~ N(0, sigma^2),
    where sigma^2 is estimated as sample variance.

    Parameters
    ----------
    residuals : np.ndarray
        Array of model residuals.

    Returns
    -------
    float
        Log-likelihood value.
    """

    if not isinstance(residuals, np.ndarray):
        raise TypeError("Residuals must be a numpy array.")

    if residuals.ndim != 1:
        raise ValueError("Residuals must be 1-dimensional.")
    
    n = len(residuals)
    if n == 0:
        raise ValueError("Residual array is empty.")
    
    sigma2 = np.var(residuals, ddof=0)

    if sigma2 <= 0:
        raise ValueError("Residual variance must be positive.")

    loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) +1)

    return float(loglik)

def student_t_loglik(residuals: np.ndarray, df: float) -> float:
    """
    Compute Student-t log-likelihood for residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    df : float
        Degrees of freedom (must be > 2).

    Returns
    -------
    float
        Log-likelihood value.
    """
    if not isinstance(residuals, np.ndarray):
        raise TypeError("Residuals must be a numpy array.")

    if residuals.ndim != 1:
        raise ValueError("Residuals must be 1-dimensional.")

    if df <= 2:
        raise ValueError("Degrees of freedom must be greater than 2.")

    n = len(residuals)
    if n == 0:
        raise ValueError("Residual array is empty.")

    sigma2 = np.var(residuals, ddof=0)

    if sigma2 <= 0:
        raise ValueError("Residual variance must be positive.")

    # Log-likelihood formula for Student-t
    term1 = gammaln((df + 1) / 2)
    term2 = gammaln(df / 2)
    term3 = 0.5 * np.log((df - 2) * np.pi * sigma2)

    standardized = residuals**2 / ((df - 2) * sigma2)
    term4 = ((df + 1) / 2) * np.log(1 + standardized)

    loglik = n * (term1 - term2 - term3) - np.sum(term4)

    return float(loglik)

