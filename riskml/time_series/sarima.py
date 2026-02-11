"""
SARIMA(p,d,q)(P,D,Q,s) model
"""

import numpy as np
from riskml.time_series.arima import ARIMA


class SARIMA(ARIMA):
    """
    Seasonal ARIMA model.
    """
    def __init__(self, p: int, d: int, q: int, P: int, D: int, Q: int, s: int):
        super().__init__(p, d, q)
        if P < 0 or D < 0 or Q < 0 or s < 1:
            raise ValueError("Seasonal parameters must be valid")
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.seasonal_model: ARIMA = None

    def fit(self, y: np.ndarray) -> "SARIMA":
        self._validate_input(y)
        self.n_obs_ = len(y)

        # Apply seasonal differencing
        if self.D > 0:
            y_seasonal_diff = y[self.s * self.D :] - y[: -self.s * self.D]
        else:
            y_seasonal_diff = y

        # Fit ARIMA on seasonally differenced series
        self.seasonal_model = ARIMA(self.p, self.d, self.q)
        self.seasonal_model.fit(y_seasonal_diff)
        self.params_ = self.seasonal_model.params_
        self.residuals_ = self.seasonal_model.residuals_
        self.intercept_ = self.seasonal_model.intercept_
        self.ar_coefs_ = self.seasonal_model.ar_coefs_
        self.ma_coefs_ = self.seasonal_model.ma_coefs_
        self.fitted_ = True
        self._post_fit_statistics(len(self.params_))
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        diff_forecast = self.seasonal_model.predict(steps)
        return diff_forecast  # Simple placeholder; seasonal back-transform can be added

    def simulate(self, n: int) -> np.ndarray:
        sim = self.seasonal_model.simulate(n)
        return sim
