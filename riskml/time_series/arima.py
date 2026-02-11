"""
ARIMA(p,d,q) model
"""

import numpy as np
from riskml.time_series.arma import ARMA
from riskml.time_series.base import BaseTimeSeriesModel


class ARIMA(BaseTimeSeriesModel):
    """
    ARIMA(p,d,q) model using ARMA on differenced data.
    """
    def __init__(self, p: int, d: int, q: int):
        super().__init__()
        if p < 0 or d < 0 or q < 0:
            raise ValueError("p, d, q must be >= 0")
        self.p = p
        self.d = d
        self.q = q
        self.arma_model: ARMA = ARMA(p, q)
        self.history_: np.ndarray = None

    def fit(self, y: np.ndarray) -> "ARIMA":
        self._validate_input(y)
        self.n_obs_ = len(y)

        # Difference the series
        self.history_ = y.copy()
        y_diff = np.diff(y, n=self.d)

        self.arma_model.fit(y_diff)
        self.params_ = self.arma_model.params_
        self.residuals_ = self.arma_model.residuals_
        self.intercept_ = self.arma_model.intercept_
        self.ar_coefs_ = self.arma_model.ar_coefs_
        self.ma_coefs_ = self.arma_model.ma_coefs_
        self.fitted_ = True
        self._post_fit_statistics(len(self.params_))
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted first.")

        diff_forecast = self.arma_model.predict(steps)
        y_last = self.history_[-self.d :]
        y_forecast = np.r_[y_last, diff_forecast].cumsum()[-steps:]
        return y_forecast

    def simulate(self, n: int) -> np.ndarray:
        diff_sim = self.arma_model.simulate(n)
        y_last = self.history_[-self.d :]
        return np.r_[y_last, diff_sim].cumsum()[-n:]
