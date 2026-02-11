"""
Information criteria for model selection.

Provides reusable functions for computing AIC and BIC.
"""

from __future__ import annotations


def compute_aic(loglik: float, k: int, n: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    AIC = -2 * loglik + 2 * k

    Parameters
    ----------
    loglik : float
        Log-likelihood of model.
    k : int
        Number of estimated parameters.
    n : int
        Number of observations.

    Returns
    -------
    float
        AIC value.
    """
    if n <= 0:
        raise ValueError("Number of observations must be positive.")
    if k <= 0:
        raise ValueError("Number of parameters must be positive.")

    return -2.0 * loglik + 2.0 * k


def compute_bic(loglik: float, k: int, n: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    BIC = -2 * loglik + k * log(n)

    Parameters
    ----------
    loglik : float
        Log-likelihood of model.
    k : int
        Number of estimated parameters.
    n : int
        Number of observations.

    Returns
    -------
    float
        BIC value.
    """
    if n <= 0:
        raise ValueError("Number of observations must be positive.")
    if k <= 0:
        raise ValueError("Number of parameters must be positive.")

    import math
    return -2.0 * loglik + k * math.log(n)
