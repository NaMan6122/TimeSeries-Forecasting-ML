"""
Transformations — Week 2.

Two complementary classes for making a time series stationary:

1. VarianceStabiliser — handles heteroscedastic variance via log or Box-Cox.
   Always apply BEFORE differencing (differencing a series with growing
   variance amplifies the problem).

2. Differencer — handles trend and seasonality via first and seasonal
   differencing. Automatically determines d and D using ADF + KPSS in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import acf


class TransformMethod(Enum):
    NONE = auto()
    LOG = auto()
    BOXCOX = auto()
    SQRT = auto()


@dataclass
class TransformResult:
    """Records what VarianceStabiliser did, so it can be reversed."""

    method: TransformMethod
    lam: Optional[float]       # Box-Cox lambda (None for log/sqrt)
    original_min: float        # for shift if series had non-positive values
    shift: float               # constant added before transform

    def __repr__(self) -> str:
        if self.method == TransformMethod.LOG:
            return f"TransformResult(method=LOG, shift={self.shift:.2f})"
        if self.method == TransformMethod.BOXCOX:
            return f"TransformResult(method=BOXCOX, λ={self.lam:.4f}, shift={self.shift:.2f})"
        return f"TransformResult(method={self.method.name})"


class VarianceStabiliser:
    """
    Stabilise the variance of a time series before differencing.

    If the series has variance that grows with its level (common in economic
    and count data), a power transform is needed. This class supports:

      - log transform (λ ≈ 0): the most common and interpretable choice
      - Box-Cox with automatic λ selection: data-driven, optimal in MLE sense
      - sqrt (λ ≈ 0.5): less aggressive than log

    Parameters
    ----------
    method : TransformMethod or None
        If None, auto-selects via Box-Cox λ. If LOG/SQRT, uses that directly.
    """

    def __init__(self, method: Optional[TransformMethod] = None) -> None:
        self._method_override = method
        self._result: Optional[TransformResult] = None

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """
        Auto-select and apply the best variance-stabilising transform.

        Logic:
          1. Run Box-Cox with auto λ
          2. If λ ≈ 0 → use log (simpler, more interpretable)
          3. If λ ≈ 0.5 → use sqrt
          4. If λ ≈ 1 → no transform needed
          5. Otherwise → use Box-Cox with the fitted λ

        Returns the transformed series.
        """
        if self._method_override is not None:
            if self._method_override == TransformMethod.LOG:
                return self.log_transform(series)
            if self._method_override == TransformMethod.SQRT:
                return self._apply_sqrt(series)
            if self._method_override == TransformMethod.NONE:
                self._result = TransformResult(
                    method=TransformMethod.NONE, lam=None,
                    original_min=float(series.min()), shift=0.0,
                )
                return series

        shift = 0.0
        if series.min() <= 0:
            shift = abs(series.min()) + 1.0

        shifted = series + shift
        _, lam = sp_stats.boxcox(shifted.values)
        lam = float(lam)

        if abs(lam) < 0.1:
            return self.log_transform(series)
        if abs(lam - 0.5) < 0.15:
            return self._apply_sqrt(series)
        if abs(lam - 1.0) < 0.15:
            self._result = TransformResult(
                method=TransformMethod.NONE, lam=lam,
                original_min=float(series.min()), shift=0.0,
            )
            return series

        return self.boxcox_transform(series, lam=lam)

    def log_transform(self, series: pd.Series) -> pd.Series:
        """Apply log transform. Shifts series if it contains non-positive values."""
        shift = 0.0
        if series.min() <= 0:
            shift = abs(series.min()) + 1.0

        transformed = np.log(series + shift)
        self._result = TransformResult(
            method=TransformMethod.LOG, lam=0.0,
            original_min=float(series.min()), shift=shift,
        )
        return transformed.rename(series.name)

    def boxcox_transform(
        self, series: pd.Series, lam: Optional[float] = None
    ) -> pd.Series:
        """
        Apply Box-Cox transform.

        If lam is None, optimal λ is estimated via MLE (scipy.stats.boxcox).
        """
        shift = 0.0
        if series.min() <= 0:
            shift = abs(series.min()) + 1.0

        shifted = series + shift
        if lam is None:
            transformed_arr, lam = sp_stats.boxcox(shifted.values)
        else:
            transformed_arr = sp_stats.boxcox(shifted.values, lmbda=lam)

        self._result = TransformResult(
            method=TransformMethod.BOXCOX, lam=float(lam),
            original_min=float(series.min()), shift=shift,
        )
        return pd.Series(transformed_arr, index=series.index, name=series.name)

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Reverse the last applied transform."""
        if self._result is None:
            raise RuntimeError("No transform applied yet.")

        r = self._result
        if r.method == TransformMethod.NONE:
            return series
        if r.method == TransformMethod.LOG:
            return (np.exp(series) - r.shift).rename(series.name)
        if r.method == TransformMethod.SQRT:
            return ((series ** 2) - r.shift).rename(series.name)
        if r.method == TransformMethod.BOXCOX:
            from scipy.special import inv_boxcox
            return pd.Series(
                inv_boxcox(series.values, r.lam) - r.shift,
                index=series.index, name=series.name,
            )
        raise ValueError(f"Unknown method: {r.method}")

    @property
    def result(self) -> TransformResult:
        if self._result is None:
            raise RuntimeError("No transform applied yet.")
        return self._result

    def _apply_sqrt(self, series: pd.Series) -> pd.Series:
        shift = 0.0
        if series.min() < 0:
            shift = abs(series.min()) + 1.0
        self._result = TransformResult(
            method=TransformMethod.SQRT, lam=0.5,
            original_min=float(series.min()), shift=shift,
        )
        return np.sqrt(series + shift).rename(series.name)

    def __repr__(self) -> str:
        fitted = self._result is not None
        return f"VarianceStabiliser(fitted={fitted})"


@dataclass
class DifferencingResult:
    """Records the differencing applied so it can be reversed."""

    d: int                           # non-seasonal differencing order
    D: int                           # seasonal differencing order
    seasonal_period: int             # m
    initial_values: list[pd.Series]  # stored for inverse_difference

    def __repr__(self) -> str:
        return (
            f"DifferencingResult(d={self.d}, D={self.D}, "
            f"m={self.seasonal_period})"
        )


class Differencer:
    """
    Apply first and seasonal differencing to achieve stationarity.

    Key concepts:
      - First differencing (d): yₜ' = yₜ - yₜ₋₁. Removes linear trend.
      - Seasonal differencing (D): yₜ' = yₜ - yₜ₋ₘ. Removes seasonal pattern.
      - Over-differencing: differencing a stationary series introduces
        artificial negative autocorrelation at lag 1 (ACF₁ ≈ -0.5).

    Parameters
    ----------
    max_d : int
        Maximum non-seasonal differencing order to try. Default 2.
    max_D : int
        Maximum seasonal differencing order to try. Default 1.
    seasonal_period : int
        Seasonal period m (e.g. 12 for monthly). Default 12.
    """

    def __init__(
        self,
        max_d: int = 2,
        max_D: int = 1,
        seasonal_period: int = 12,
    ) -> None:
        self.max_d = max_d
        self.max_D = max_D
        self.seasonal_period = seasonal_period
        self._result: Optional[DifferencingResult] = None

    def fit_transform(
        self,
        series: pd.Series,
        alpha: float = 0.05,
    ) -> pd.Series:
        """
        Automatically determine d and D using ADF + KPSS in a loop,
        then apply the differencing.

        Algorithm:
          1. Test current series with ADF + KPSS
          2. If stationary → done
          3. If non-stationary → apply one difference, repeat
          4. After each step, check for over-differencing

        Returns the differenced (stationary) series.
        """
        from week__02__stationarity.stationarity_tests import StationarityTester

        tester = StationarityTester(alpha=alpha)
        current = series.copy()
        initial_values: list[pd.Series] = []
        d, D = 0, 0

        # Step 1: try seasonal differencing first if period is present
        result = tester.fit(current)
        if not result.is_stationary and self.max_D > 0:
            acf_vals = acf(current.dropna(), nlags=self.seasonal_period * 2, fft=True)
            if (len(acf_vals) > self.seasonal_period
                    and abs(acf_vals[self.seasonal_period]) > 0.5):
                initial_values.append(current.iloc[:self.seasonal_period].copy())
                current = current.diff(self.seasonal_period).dropna()
                D = 1

        # Step 2: first differencing
        for _ in range(self.max_d):
            result = tester.fit(current)
            if result.is_stationary:
                break
            if self._is_over_differenced(current):
                break
            initial_values.append(current.iloc[:1].copy())
            current = current.diff().dropna()
            d += 1

        self._result = DifferencingResult(
            d=d, D=D,
            seasonal_period=self.seasonal_period,
            initial_values=initial_values,
        )
        return current

    def difference(
        self,
        series: pd.Series,
        d: int = 1,
        D: int = 0,
        m: Optional[int] = None,
    ) -> pd.Series:
        """Apply specified differencing explicitly (no auto-detection)."""
        m = m or self.seasonal_period
        current = series.copy()
        initial_values: list[pd.Series] = []

        if D > 0:
            for _ in range(D):
                initial_values.append(current.iloc[:m].copy())
                current = current.diff(m).dropna()

        for _ in range(d):
            initial_values.append(current.iloc[:1].copy())
            current = current.diff().dropna()

        self._result = DifferencingResult(
            d=d, D=D,
            seasonal_period=m,
            initial_values=initial_values,
        )
        return current

    @staticmethod
    def is_over_differenced(series: pd.Series) -> bool:
        """
        Check for over-differencing using ACF at lag 1.

        Heuristic: if ACF(1) of the differenced series is strongly negative
        (< -0.5), the series was over-differenced.
        """
        return Differencer._is_over_differenced(series)

    @staticmethod
    def _is_over_differenced(series: pd.Series) -> bool:
        arr = np.asarray(series.dropna(), dtype=np.float64)
        if len(arr) < 5:
            return False
        acf_vals = acf(arr, nlags=1, fft=True)
        return float(acf_vals[1]) < -0.5

    @property
    def result(self) -> DifferencingResult:
        if self._result is None:
            raise RuntimeError("No differencing applied yet.")
        return self._result

    def __repr__(self) -> str:
        fitted = self._result is not None
        return (
            f"Differencer(max_d={self.max_d}, max_D={self.max_D}, "
            f"m={self.seasonal_period}, fitted={fitted})"
        )
