"""
TimeSeriesDecomposer — Week 1.

Wraps statsmodels STL and classical decomposition in a class that:
  - decides additive vs multiplicative from data characteristics
  - holds all three components (trend, seasonal, residual) as named Series
  - exposes residual diagnostics (stationarity of Rₜ, variance check)
  - produces a clean summary of what the decomposition revealed
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose

class DecompositionModel(Enum):
    ADDITIVE = auto()        # y_t = T_t + S_t + R_t
    MULTIPLICATIVE = auto()  # y_t = T_t * S_t * R_t

@dataclass
class DecompositionResult:
    """
    Holds the three components of a decomposition.

    Attributes
    ----------
    trend    : Tₜ — the slow-moving level of the series
    seasonal : Sₜ — the repeating periodic pattern
    residual : Rₜ — what remains after removing T and S
    model    : which model was used (additive / multiplicative)
    period   : the seasonal period assumed (e.g. 12 for monthly)
    """

    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    model: DecompositionModel
    period: int

    @property
    def seasonal_strength(self) -> float:
        """
        Ratio of seasonal variance to total variance (after removing trend).

        A value close to 1 means seasonality dominates.
        A value near 0 means the series is dominated by trend + noise.

        Formula: 1 - Var(Rₜ) / Var(Sₜ + Rₜ)
        """
        sr = self.seasonal.dropna() + self.residual.dropna()
        if sr.var() == 0:
            return 0.0
        return float(max(0.0, 1.0 - self.residual.dropna().var() / sr.var()))

    @property
    def trend_strength(self) -> float:
        """
        Ratio of trend variance to total variance (after removing seasonal).

        Formula: 1 - Var(Rₜ) / Var(Tₜ + Rₜ)
        """
        tr = self.trend.dropna() + self.residual.dropna()
        if tr.var() == 0:
            return 0.0
        return float(max(0.0, 1.0 - self.residual.dropna().var() / tr.var()))

    def __repr__(self) -> str:
        return (
            f"DecompositionResult("
            f"model={self.model.name}, "
            f"period={self.period}, "
            f"seasonal_strength={self.seasonal_strength:.3f}, "
            f"trend_strength={self.trend_strength:.3f})"
        )

class TimeSeriesDecomposer:
    """
    Decompose a time series into trend, seasonal, and residual components.

    Supports two methods:
      - STL (Seasonal-Trend decomposition using LOESS) — the recommended
        default. Unlike classical decomposition, STL allows the seasonal
        shape to evolve slowly over time.
      - Classical (statsmodels seasonal_decompose) — simpler, faster,
        but seasonal shape is fixed.

    Parameters
    ----------
    period : int or None
        Seasonal period (e.g. 12 for monthly data with annual seasonality,
        4 for quarterly). If None, inferred from the index frequency.

    model : DecompositionModel or None
        If None, the model is auto-selected by inspecting whether seasonal
        amplitude grows with the series level (→ multiplicative) or stays
        constant (→ additive).
    """

    def __init__(
        self,
        period: Optional[int] = None,
        model: Optional[DecompositionModel] = None,
    ) -> None:
        self.period = period
        self._model_override = model
        self._result: Optional[DecompositionResult] = None

    def fit_stl(
        self,
        series: pd.Series,
        robust: bool = True,
        seasonal: int = 7,
    ) -> DecompositionResult:
        """
        Fit STL decomposition.

        STL uses locally weighted regression (LOESS) to estimate trend
        and seasonal components iteratively. The key advantage over
        classical decomposition is that the seasonal component can
        change shape over time — useful for real-world data.

        Parameters
        ----------
        series   : univariate time series with DatetimeIndex
        robust   : if True, use robust LOESS iteration to down-weight
                   outliers. Recommended for most real data.
        seasonal : smoothing window for the seasonal component (must be
                   odd). Larger = smoother seasonal estimate.

        Returns
        -------
        DecompositionResult
        """
        period = self._resolve_period(series)
        model = self._select_model(series)

        work_series = np.log(series) if model == DecompositionModel.MULTIPLICATIVE else series

        stl = STL(work_series, period=period, seasonal=seasonal, robust=robust)
        fit = stl.fit()

        if model == DecompositionModel.MULTIPLICATIVE:
            trend    = np.exp(pd.Series(fit.trend,    index=series.index, name="trend"))
            seasonal = np.exp(pd.Series(fit.seasonal, index=series.index, name="seasonal"))
            residual = series / (trend * seasonal)
            residual.name = "residual"
        else:
            trend    = pd.Series(fit.trend,    index=series.index, name="trend")
            seasonal = pd.Series(fit.seasonal, index=series.index, name="seasonal")
            residual = pd.Series(fit.resid,    index=series.index, name="residual")

        self._result = DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            model=model,
            period=period,
        )
        return self._result

    def fit_classical(self, series: pd.Series) -> DecompositionResult:
        """
        Fit classical (moving-average based) decomposition.

        Simpler than STL — uses a centred moving average of length=period
        to estimate trend, then averages residuals by position within the
        cycle to estimate seasonality. The seasonal component is fixed
        (same shape every period).
        """
        period = self._resolve_period(series)
        model = self._select_model(series)
        model_str = "additive" if model == DecompositionModel.ADDITIVE else "multiplicative"

        result = seasonal_decompose(series, model=model_str, period=period, extrapolate_trend=1)

        self._result = DecompositionResult(
            trend    = result.trend.rename("trend"),
            seasonal = result.seasonal.rename("seasonal"),
            residual = result.resid.rename("residual"),
            model    = model,
            period   = period,
        )
        return self._result

    @property
    def result(self) -> DecompositionResult:
        self._assert_fitted()
        return self._result

    def residual_diagnostics(self) -> dict[str, float | bool]:
        """
        Basic checks on the residual component Rₜ.

        For a well-specified decomposition, Rₜ should be:
          - stationary (constant mean and variance over time)
          - near-normal
          - not autocorrelated

        Returns a dict of diagnostic values for quick inspection.
        """
        from scipy import stats
        from statsmodels.stats.stattools import durbin_watson

        self._assert_fitted()
        r = self._result.residual.dropna()

        _, shapiro_p = stats.shapiro(r)
        dw = durbin_watson(r.values)

        # Rough stationarity check: variance in first half vs second half
        mid = len(r) // 2
        var_ratio = float(r.iloc[mid:].var() / r.iloc[:mid].var()) if r.iloc[:mid].var() > 0 else np.nan

        return {
            "mean": float(r.mean()),
            "std": float(r.std()),
            "shapiro_p": float(shapiro_p),
            "normality_ok": shapiro_p > 0.05,
            "durbin_watson": float(dw),
            # DW ≈ 2 → no autocorrelation, < 1 or > 3 → concern
            "dw_ok": 1.0 < dw < 3.0,
            "variance_ratio_halves": var_ratio,
            # ratio close to 1 → homoscedastic residuals, refer notes.
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_period(self, series: pd.Series) -> int:
        if self.period is not None:
            return self.period

        freq = getattr(series.index, "freqstr", None) or pd.infer_freq(series.index)
        freq_to_period = {
            "MS": 12, "M": 12, "ME": 12,
            "QS": 4,  "Q": 4,  "QE": 4,
            "W": 52,
            "D": 7,
            "H": 24,
            "A": 1, "AS": 1, "YE": 1,
        }
        if freq and freq.upper() in freq_to_period:
            return freq_to_period[freq.upper()]
        
        # Trailing suffix trim
        for key, val in freq_to_period.items():
            if freq and freq.upper().startswith(key):
                return val

        raise ValueError(
            f"Cannot infer seasonal period from frequency '{freq}'. "
            "Pass period= explicitly."
        )

    def _select_model(self, series: pd.Series) -> DecompositionModel:
        """
        Auto-select additive vs multiplicative by comparing seasonal
        amplitude to the local level.

        Method: split the series into rolling windows and measure whether
        the range (max-min) within each window scales with the window mean.
        A strong positive correlation → multiplicative.
        """
        if self._model_override is not None:
            return self._model_override

        period = self._resolve_period(series)
        # Rolling range and mean over each seasonal window
        rolling_range = series.rolling(period).apply(lambda x: x.max() - x.min())
        rolling_mean  = series.rolling(period).mean()

        valid = ~(rolling_range.isna() | rolling_mean.isna())
        if valid.sum() < 2:
            return DecompositionModel.ADDITIVE

        from scipy.stats import pearsonr
        corr, _ = pearsonr(rolling_mean[valid], rolling_range[valid])

        # Threshold 0.7: correlation above this suggests the amplitude
        # grows with the level → multiplicative structure
        return (
            DecompositionModel.MULTIPLICATIVE
            if corr > 0.7
            else DecompositionModel.ADDITIVE
        )

    def _assert_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("No decomposition fitted yet. Call fit_stl() or fit_classical().")

    def __repr__(self) -> str:
        fitted = self._result is not None
        return (
            f"TimeSeriesDecomposer("
            f"period={self.period}, "
            f"fitted={fitted})"
        )
