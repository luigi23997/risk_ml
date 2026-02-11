"""
Autoregressive model AR(p)
"""

from typing import Optional
import numpy as np

from riskml.time_series.arma_base import BaseARMA
from riskml.core.linear_model import ols


class AR(BaseARMA):
    """
    AR(p) model using OLS.
    """
    def __init__(self, p: int):
        super().__init__()
        if p < 1:
            raise ValueError("p must be >= 1")
        self.p = p
        self.ar_coefs_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, y: np.ndarray) -> "AR":
        self._validate_input(y)
        self.n_obs_ = len(y)

        # Build lag matrix
        X = np.ones((len(y) - self.p, self.p + 1))
        for i in range(self.p):
            X[:, i + 1] = y[self.p - i - 1 : len(y) - i - 1]
        y_target = y[self.p:]

        # OLS
        beta = ols(X, y_target)
        self.intercept_ = beta[0]
        self.ar_coefs_ = beta[1:]
        self.params_ = beta
        self.residuals_ = y_target - X @ beta
        self.fitted_ = True

        # Statistics
        self._post_fit_statistics(len(beta))
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        history = list(self._get_history())
        forecasts = []
        for _ in range(steps):
            y_pred = self.intercept_
            for i in range(self.p):
                y_pred += self.ar_coefs_[i] * history[-i - 1]
            forecasts.append(y_pred)
            history.append(y_pred)

        return np.array(forecasts)

    def _get_history(self):
        """
        Returns full history for prediction
        """
        return list(self.residuals_ + self.intercept_)

    def simulate(self, n: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        y = list(self._get_history())
        for _ in range(n):
            y_pred = self.intercept_ + sum(
                self.ar_coefs_[i] * y[-i - 1] for i in range(self.p)
            )
            y.append(y_pred + np.random.normal(0, np.std(self.residuals_)))
        return np.array(y[-n:])
