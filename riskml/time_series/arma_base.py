"""
Base class for ARMA-family models.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from riskml.time_series.base import BaseTimeSeriesModel
from riskml.core.likelihood import gaussian_loglik
from riskml.time_series.information_criteria import compute_aic, compute_bic


class BaseARMA(BaseTimeSeriesModel):
    """
    Base class for ARMA-type models (ARMA, ARIMA, SARIMA)
    """
    def __init__(self):
        super().__init__()
        self.ar_coefs_: Optional[np.ndarray] = None
        self.ma_coefs_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    # =====================================================
    # Residual recursion
    # =====================================================
    def _compute_residuals(self, params: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute residuals recursively. Expects AR and MA parameters.
        This method can be overridden for AR or MA shortcuts.
        """
        p = 0 if self.ar_coefs_ is None else len(self.ar_coefs_)
        q = 0 if self.ma_coefs_ is None else len(self.ma_coefs_)

        n = len(y)
        residuals = np.zeros(n)

        # Intercept
        intercept = params[0]

        ar_params = params[1 : 1 + p] if p > 0 else np.array([])
        ma_params = params[1 + p : 1 + p + q] if q > 0 else np.array([])

        for t in range(n):
            ar_term = sum(ar_params[i] * y[t - i - 1] for i in range(min(p, t)))
            ma_term = sum(ma_params[i] * residuals[t - i - 1] for i in range(min(q, t)))
            residuals[t] = y[t] - (intercept + ar_term + ma_term)

        return residuals

    # =====================================================
    # Post-fit statistics
    # =====================================================
    def _post_fit_statistics(self, k: int):
        """
        Compute log-likelihood, AIC, BIC after fitting.
        """
        self.loglik_ = gaussian_loglik(self.residuals_)
        self.aic_ = compute_aic(self.loglik_, k, self.n_obs_)
        self.bic_ = compute_bic(self.loglik_, k, self.n_obs_)

    # =====================================================
    # Stability checks
    # =====================================================
    def _check_stationarity(self):
        if self.ar_coefs_ is None or len(self.ar_coefs_) == 0:
            return True
        # AR polynomial roots must be outside unit circle
        ar_poly = np.r_[1, -self.ar_coefs_]
        roots = np.roots(ar_poly)
        return np.all(np.abs(roots) > 1.0)

    def _check_invertibility(self):
        if self.ma_coefs_ is None or len(self.ma_coefs_) == 0:
            return True
        ma_poly = np.r_[1, self.ma_coefs_]
        roots = np.roots(ma_poly)
        return np.all(np.abs(roots) > 1.0)
