"""
AutocorrelationAnalyser.

Encapsulates ACF and PACF computation and interpretation.

Key insight encoded here:
  ACF at lag k = total correlation between yₜ and yₜ₋ₖ
               = direct + indirect (via shorter lags)
  PACF at lag k = only the *direct* correlation,
                  after partialling out lags 1..k-1

    AR(p) → PACF cuts off at p, ACF tails off
    MA(q) → ACF cuts off at q, PACF tails off
    ARMA  → both tail off
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

class SuggestedProcess(Enum):
    AR = auto()
    MA = auto()
    ARMA = auto()
    WHITE_NOISE = auto()
    UNKNOWN = auto()


@dataclass
class ACFPACFResult:
    """
    Stores ACF and PACF values together with their confidence bands.

    Attributes
    ----------
    acf_values  : autocorrelation at each lag (lag 0 = 1.0 always)
    pacf_values : partial autocorrelation at each lag
    acf_conf    : 95% confidence interval array, shape (n_lags, 2)
    pacf_conf   : 95% confidence interval array, shape (n_lags, 2)
    n_lags      : number of lags computed
    n_obs       : length of the original series
    """

    acf_values: npt.NDArray[np.float64]
    pacf_values: npt.NDArray[np.float64]
    acf_conf: npt.NDArray[np.float64]
    pacf_conf: npt.NDArray[np.float64]
    n_lags: int
    n_obs: int

    @property
    def significant_acf_lags(self) -> list[int]:
        """
        Lags (excluding 0) where ACF is statistically significant.

        statsmodels returns confidence intervals centred on the estimate
        itself, so the test is: does the interval NOT contain zero?
        """
        return [
            k for k in range(1, self.n_lags + 1)
            if not (self.acf_conf[k, 0] <= 0 <= self.acf_conf[k, 1])
        ]

    @property
    def significant_pacf_lags(self) -> list[int]:
        """
        Lags (excluding 0) where PACF is statistically significant.

        Same logic as significant_acf_lags — CI is centred on the value.
        """
        return [
            k for k in range(1, self.n_lags + 1)
            if not (self.pacf_conf[k, 0] <= 0 <= self.pacf_conf[k, 1])
        ]

    def __repr__(self) -> str:
        return (
            f"ACFPACFResult("
            f"n_obs={self.n_obs}, n_lags={self.n_lags}, "
            f"sig_acf={self.significant_acf_lags[:5]}, "
            f"sig_pacf={self.significant_pacf_lags[:5]})"
        )

class AutocorrelationAnalyser:
    """
    Compute and interpret ACF and PACF for a univariate time series.

    Parameters
    ----------
    n_lags : int
        Number of lags to compute. Default = min(40, n // 2 - 1).
    alpha : float
        Significance level for the confidence bands. Default 0.05 (95%).

    Examples
    --------
    >>> analyser = AutocorrelationAnalyser(n_lags=24)
    >>> result = analyser.fit(series)
    >>> print(analyser.interpret())
    """

    def __init__(
        self,
        n_lags: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        self.n_lags = n_lags
        self.alpha = alpha
        self._result: Optional[ACFPACFResult] = None
        self._series_name: str = "series"

    def fit(self, series: pd.Series) -> ACFPACFResult:
        """
        Compute ACF and PACF for the given series.

        Important prerequisite: the series should be stationary before
        calling this. ACF/PACF are not meaningful on a trending series
        because slow ACF decay simply reflects the trend, not genuine
        autocorrelation structure in the errors.

        Parameters
        ----------
        series : univariate pd.Series

        Returns
        -------
        ACFPACFResult
        """
        arr = np.asarray(series.dropna(), dtype=np.float64)
        n = len(arr)
        n_lags = self.n_lags or min(40, n // 2 - 1)

        acf_vals, acf_conf = acf(arr, nlags=n_lags, alpha=self.alpha, fft=True)

        # PACF method 'ywm' (Yule-Walker with bias correction) is more
        # stable than OLS for most real time series
        pacf_vals, pacf_conf = pacf(arr, nlags=n_lags, alpha=self.alpha, method="ywm")

        self._series_name = series.name or "series"
        self._result = ACFPACFResult(
            acf_values=acf_vals,
            pacf_values=pacf_vals,
            acf_conf=acf_conf,
            pacf_conf=pacf_conf,
            n_lags=n_lags,
            n_obs=n,
        )
        return self._result

    def interpret(self) -> dict[str, object]:
        """
        Apply the pattern-matching rules to suggest a model family.

        Returns a dictionary with:
          - suggested_process : SuggestedProcess enum value
          - suggested_ar_order: integer p (or None)
          - suggested_ma_order: integer q (or None)
          - reasoning          : human-readable explanation
        """
        self._assert_fitted()
        r = self._result

        sig_acf = r.significant_acf_lags
        sig_pacf = r.significant_pacf_lags

        if not sig_acf and not sig_pacf:
            return {
                "suggested_process": SuggestedProcess.WHITE_NOISE,
                "suggested_ar_order": 0,
                "suggested_ma_order": 0,
                "reasoning": (
                    "No significant lags in ACF or PACF. "
                    "The series appears to be white noise — no autocorrelation to model."
                ),
            }

        acf_cuts_off  = self._appears_to_cut_off(sig_acf,  r.acf_values,  r.n_lags)
        pacf_cuts_off = self._appears_to_cut_off(sig_pacf, r.pacf_values, r.n_lags)

        if pacf_cuts_off and not acf_cuts_off:
            p = max(sig_pacf) if sig_pacf else 0
            return {
                "suggested_process": SuggestedProcess.AR,
                "suggested_ar_order": p,
                "suggested_ma_order": 0,
                "reasoning": (
                    f"PACF has significant spikes up to lag {p} then cuts off. "
                    f"ACF decays gradually. Pattern suggests AR({p})."
                ),
            }

        if acf_cuts_off and not pacf_cuts_off:
            q = max(sig_acf) if sig_acf else 0
            return {
                "suggested_process": SuggestedProcess.MA,
                "suggested_ar_order": 0,
                "suggested_ma_order": q,
                "reasoning": (
                    f"ACF has significant spikes up to lag {q} then cuts off. "
                    f"PACF decays gradually. Pattern suggests MA({q})."
                ),
            }

        p_hint = min(sig_pacf[:3]) if sig_pacf else 1
        q_hint = min(sig_acf[:3]) if sig_acf else 1
        return {
            "suggested_process": SuggestedProcess.ARMA,
            "suggested_ar_order": p_hint,
            "suggested_ma_order": q_hint,
            "reasoning": (
                "Both ACF and PACF tail off without a sharp cutoff. "
                "Pattern suggests ARMA — use AIC/BIC grid search to determine p and q. "
                f"Starting hints: p≈{p_hint}, q≈{q_hint}."
            ),
        }

    @property
    def result(self) -> ACFPACFResult:
        self._assert_fitted()
        return self._result

    @staticmethod
    def _appears_to_cut_off(
        significant_lags: list[int],
        values: npt.NDArray[np.float64],
        n_lags: int,
    ) -> bool:
        """
        Determine whether a plot 'cuts off' (sharp boundary) vs 'tails off'
        (gradual decay to zero).

        """
        if not significant_lags:
            return True

        max_sig = max(significant_lags)

        if max_sig <= 2 and len(significant_lags) <= 2:
            return True

        if len(values) > 5:
            v1 = abs(values[1])
            v4 = abs(values[min(4, len(values) - 1)])
            if v1 > 0.05:
                decay_ratio = v4 / v1
                if decay_ratio > 0.2:
                    return False

        return max_sig <= 3

    def _assert_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError(
                "No series fitted yet. Call .fit(series) first."
            )

    def __repr__(self) -> str:
        fitted = self._result is not None
        return (
            f"AutocorrelationAnalyser("
            f"n_lags={self.n_lags}, alpha={self.alpha}, fitted={fitted})"
        )
