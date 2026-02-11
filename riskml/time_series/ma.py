"""
Moving Average model MA(q)
"""

from typing import Optional
import numpy as np
from scipy.optimize import minimize

from riskml.time_series.arma_base import BaseARMA
from riskml.core.likelihood import gaussian_loglik


class MA(BaseARMA):
    """
    MA(q) model using maximum likelihood estimation.
    """
    def __init__(self, q: int):
        super().__init__()
        if q < 1:
            raise ValueError("q must be >= 1")
        self.q = q
        self.ma_coefs_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "MA":
        self._validate_input(y)
        self.n_obs_ = len(y)

        # Initial guess
        initial_params = np.zeros(self.q + 1)  # intercept + MA coefficients

        # Negative log-likelihood
        def neg_loglik(params):
            residuals = self._compute_residuals(params, y)
            return -1.0 * gaussian_loglik(residuals)

        res = minimize(neg_loglik, initial_params, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError("MA fitting failed: " + res.message)

        self.params_ = res.x
        self.intercept_ = res.x[0]
        self.ma_coefs_ = res.x[1:]
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
            ma_term = sum(self.ma_coefs_[i] * history[-i - 1] for i in range(min(self.q, len(history))))
            y_pred = self.intercept_ + ma_term
            forecasts.append(y_pred)
            history.append(y_pred)

        return np.array(forecasts)

    def simulate(self, n: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        y = list(self.residuals_ + self.intercept_)
        for _ in range(n):
            ma_term = sum(self.ma_coefs_[i] * y[-i - 1] for i in range(min(self.q, len(y))))
            y_pred = self.intercept_ + ma_term + np.random.normal(0, np.std(self.residuals_))
            y.append(y_pred)
        return np.array(y[-n:])
