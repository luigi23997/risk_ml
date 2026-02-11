"""
Base classes for time series models.

All deterministic time-series models (AR, MA, ARMA)
must inherit from BaseTimeSeriesModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from riskml.core.likelihood import gaussian_loglik
from riskml.time_series.information_criteria import (
    compute_aic,
    compute_bic,
)


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for deterministic time series models.

    All models must implement:
        - fit()
        - predict()
        - simulate()
        - _compute_residuals()

    Attributes
    ----------
    params_ : np.ndarray
        Estimated model parameters.
    residuals_ : np.ndarray
        Model residuals.
    loglik_ : float
        Log-likelihood of fitted model.
    aic_ : float
        Akaike Information Criterion.
    bic_ : float
        Bayesian Information Criterion.
    n_obs_ : int
        Number of observations used in fitting.
    fitted_ : bool
        Whether model has been fitted.
    """

    def __init__(self) -> None:
        self.params_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.loglik_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.n_obs_: Optional[int] = None
        self.fitted_: bool = False

    # =====================================================
    # Abstract API
    # =====================================================

    @abstractmethod
    def fit(self, y: np.ndarray) -> "BaseTimeSeriesModel":
        """
        Fit model to time series data.

        Parameters
        ----------
        y : np.ndarray
            1D array of time series observations.

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecast values.
        """
        pass

    @abstractmethod
    def simulate(self, n: int) -> np.ndarray:
        """
        Simulate a time series from the fitted model.

        Parameters
        ----------
        n : int
            Number of observations to simulate.

        Returns
        -------
        np.ndarray
        """
        pass

    @abstractmethod
    def _compute_residuals(
        self,
        params: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute model residuals given parameters.

        This method must be implemented in subclasses.
        """
        pass

    # =====================================================
    # Shared Utilities
    # =====================================================

    def _validate_input(self, y: np.ndarray) -> None:
        """
        Validate input time series.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if y.ndim != 1:
            raise ValueError("Time series must be 1-dimensional.")

        if len(y) < 3:
            raise ValueError("Time series is too short.")

        if np.any(np.isnan(y)):
            raise ValueError("Time series contains NaN values.")

    def _post_fit_statistics(self, k: int) -> None:
        """
        Compute log-likelihood and information criteria.

        Parameters
        ----------
        k : int
            Number of estimated parameters.
        """
        if self.residuals_ is None:
            raise RuntimeError("Residuals must be computed before statistics.")

        self.loglik_ = gaussian_loglik(self.residuals_)
        self.aic_ = compute_aic(self.loglik_, k, self.n_obs_)
        self.bic_ = compute_bic(self.loglik_, k, self.n_obs_)

    def summary(self) -> str:
        """
        Return formatted model summary.
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before summary.")

        lines = []
        lines.append("=" * 50)
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append("-" * 50)

        if self.params_ is not None:
            for i, param in enumerate(self.params_):
                lines.append(f"param_{i}: {param:.6f}")

        lines.append("-" * 50)
        lines.append(f"Log-Likelihood: {self.loglik_:.4f}")
        lines.append(f"AIC:            {self.aic_:.4f}")
        lines.append(f"BIC:            {self.bic_:.4f}")
        lines.append(f"Observations:   {self.n_obs_}")
        lines.append("=" * 50)

        return "\n".join(lines)

    def get_params(self) -> np.ndarray:
        """
        Return fitted parameters.
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before accessing parameters.")
        return self.params_

    def is_fitted(self) -> bool:
        """
        Check whether model is fitted.
        """
        return self.fitted_
