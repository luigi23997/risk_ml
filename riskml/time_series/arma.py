"""
ARMA(p,q) model
"""

import numpy as np
from scipy.optimize import minimize

from riskml.time_series.arma_base import BaseARMA
from riskml.core.likelihood import gaussian_loglik


class ARMA(BaseARMA):
    """
    ARMA(p,q) model using maximum likelihood estimation.
    """
    def __init__(self, p: int, q: int):
        super().__init__()
        if p < 0 or q < 0:
            raise ValueError("p and q must be >= 0")
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be > 0")
        self.p = p
        self.q = q

    def fit(self, y: np.ndarray) -> "ARMA":
        self._validate_input(y)
        self.n_obs_ = len(y)

        # Initial guess: intercept + AR + MA
        initial_params = np.zeros(1 + self.p + self.q)
        initial_params[0] = np.mean(y)

        # Negative log-likelihood
        def neg_loglik(params):
            residuals = self._compute_residuals(params, y)
            return -1.0 * gaussian_loglik(residuals)

        res = minimize(neg_loglik, initial_params, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError("ARMA fitting failed: " + res.message)

        self.params_ = res.x
        self.intercept_ = res.x[0]
        self.ar_coefs_ = res.x[1 : 1 + self.p] if self.p > 0 else np.array([])
        self.ma_coefs_ = res.x[1 + self.p :] if self.q > 0 else np.array([])
        self.residuals_ = self._compute_residuals(res.x, y)
        self.fitted_ = True
        self._post_fit_statistics(len(res.x))
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        history = list(self.residuals_ + self.intercept_)
        forecasts = []

        for _ in range(steps):
            ar_term = sum(self.ar_coefs_[i] * history[-i - 1] for i in range(min(self.p, len(history))))
            ma_term = sum(self.ma_coefs_[i] * history[-i - 1] for i in range(min(self.q, len(history))))
            y_pred = self.intercept_ + ar_term + ma_term
            forecasts.append(y_pred)
            history.append(y_pred)

        return np.array(forecasts)

    def simulate(self, n: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        y = list(self.residuals_ + self.intercept_)
        for _ in range(n):
            ar_term = sum(self.ar_coefs_[i] * y[-i - 1] for i in range(min(self.p, len(y))))
            ma_term = sum(self.ma_coefs_[i] * y[-i - 1] for i in range(min(self.q, len(y))))
            y_pred = self.intercept_ + ar_term + ma_term + np.random.normal(0, np.std(self.residuals_))
            y.append(y_pred)
        return np.array(y[-n:])
