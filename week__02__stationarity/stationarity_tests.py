"""
StationarityTester — Week 2.

Provides a unified interface for testing whether a time series is stationary
using ADF, KPSS, and Phillips–Perron tests. The key design insight is that
ADF and KPSS have *opposite* null hypotheses:

    ADF  H₀: unit root exists   (non-stationary)
    KPSS H₀: series is stationary

Combining both gives a four-quadrant decision matrix that is more reliable
than either test alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


class StationarityVerdict(Enum):
    """Outcome of the ADF + KPSS four-quadrant decision matrix."""

    STATIONARY = auto()       # ADF rejects, KPSS fails to reject
    NON_STATIONARY = auto()   # ADF fails to reject, KPSS rejects
    TREND_STATIONARY = auto() # ADF rejects, KPSS rejects — detrend, don't difference
    INCONCLUSIVE = auto()     # ADF fails to reject, KPSS fails to reject


@dataclass
class TestResult:
    """Result of a single unit-root / stationarity test."""

    test_name: str
    statistic: float
    p_value: float
    critical_values: dict[str, float]
    null_hypothesis: str
    reject_null: bool

    def __repr__(self) -> str:
        decision = "REJECT H₀" if self.reject_null else "FAIL TO REJECT H₀"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} → {decision}"
        )


@dataclass
class StationarityResult:
    """
    Combined result from ADF + KPSS (and optionally PP).

    Attributes
    ----------
    adf     : ADF test result
    kpss    : KPSS test result
    pp      : Phillips–Perron result (None if not run)
    verdict : four-quadrant decision
    """

    adf: TestResult
    kpss: TestResult
    pp: Optional[TestResult] = None
    verdict: StationarityVerdict = StationarityVerdict.INCONCLUSIVE

    @property
    def is_stationary(self) -> bool:
        return self.verdict == StationarityVerdict.STATIONARY

    @property
    def needs_differencing(self) -> bool:
        return self.verdict == StationarityVerdict.NON_STATIONARY

    @property
    def verdict_text(self) -> str:
        messages = {
            StationarityVerdict.STATIONARY: (
                "Both tests agree: series is stationary. No differencing needed."
            ),
            StationarityVerdict.NON_STATIONARY: (
                "Both tests agree: series is non-stationary. Differencing required."
            ),
            StationarityVerdict.TREND_STATIONARY: (
                "ADF rejects unit root but KPSS rejects stationarity. "
                "Series is likely trend-stationary — detrend rather than difference."
            ),
            StationarityVerdict.INCONCLUSIVE: (
                "ADF fails to reject unit root and KPSS fails to reject stationarity. "
                "Inconclusive — inspect the series visually and consider structural breaks."
            ),
        }
        return messages[self.verdict]

    def __repr__(self) -> str:
        lines = [
            f"StationarityResult(verdict={self.verdict.name})",
            f"  {self.adf}",
            f"  {self.kpss}",
        ]
        if self.pp is not None:
            lines.append(f"  {self.pp}")
        lines.append(f"  → {self.verdict_text}")
        return "\n".join(lines)


class StationarityTester:
    """
    Test a univariate time series for stationarity using ADF + KPSS.

    Parameters
    ----------
    alpha : float
        Significance level for all tests. Default 0.05.
    kpss_regression : str
        'c' (level stationarity) or 'ct' (trend stationarity) for KPSS.
        Default 'c' — the standard choice for most time series work.
    adf_regression : str
        'c' (constant), 'ct' (constant + trend), 'n' (none) for ADF.
        Default 'c'.

    Examples
    --------
    >>> tester = StationarityTester()
    >>> result = tester.fit(series)
    >>> print(result.verdict_text)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        kpss_regression: str = "c",
        adf_regression: str = "c",
    ) -> None:
        self.alpha = alpha
        self.kpss_regression = kpss_regression
        self.adf_regression = adf_regression
        self._result: Optional[StationarityResult] = None

    def fit(
        self,
        series: pd.Series,
        run_pp: bool = False,
    ) -> StationarityResult:
        """
        Run ADF and KPSS tests, combine into a four-quadrant verdict.

        Parameters
        ----------
        series : univariate time series (numeric, no NaNs expected)
        run_pp : if True, also run the Phillips–Perron test

        Returns
        -------
        StationarityResult
        """
        arr = np.asarray(series.dropna(), dtype=np.float64)

        adf_result = self._run_adf(arr)
        kpss_result = self._run_kpss(arr)
        pp_result = self._run_pp(arr) if run_pp else None

        verdict = self._combine_verdicts(adf_result, kpss_result)

        self._result = StationarityResult(
            adf=adf_result,
            kpss=kpss_result,
            pp=pp_result,
            verdict=verdict,
        )
        return self._result

    @property
    def result(self) -> StationarityResult:
        if self._result is None:
            raise RuntimeError("No test run yet. Call .fit(series) first.")
        return self._result

    @staticmethod
    def rolling_diagnostics(
        series: pd.Series,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compute rolling mean and rolling variance for visual stationarity
        inspection.

        A stationary series should have roughly flat rolling mean and
        rolling variance over time.

        Parameters
        ----------
        series : univariate time series
        window : rolling window size. Default = seasonal period or 12.

        Returns
        -------
        DataFrame with columns: 'rolling_mean', 'rolling_var'
        """
        if window is None:
            freq = getattr(series.index, "freqstr", None) or pd.infer_freq(series.index)
            window = 12 if freq and "M" in str(freq).upper() else 12

        return pd.DataFrame({
            "rolling_mean": series.rolling(window=window).mean(),
            "rolling_var": series.rolling(window=window).var(),
        })

    # ------------------------------------------------------------------
    # Internal test runners
    # ------------------------------------------------------------------

    def _run_adf(self, arr: np.ndarray) -> TestResult:
        stat, p_value, _used_lag, _nobs, crit, _icbest = adfuller(
            arr, regression=self.adf_regression, autolag="AIC"
        )
        return TestResult(
            test_name="ADF",
            statistic=float(stat),
            p_value=float(p_value),
            critical_values={k: float(v) for k, v in crit.items()},
            null_hypothesis="Unit root exists (non-stationary)",
            reject_null=p_value < self.alpha,
        )

    def _run_kpss(self, arr: np.ndarray) -> TestResult:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, _lags, crit = kpss(arr, regression=self.kpss_regression)
        return TestResult(
            test_name="KPSS",
            statistic=float(stat),
            p_value=float(p_value),
            critical_values={k: float(v) for k, v in crit.items()},
            null_hypothesis="Series is stationary",
            reject_null=p_value < self.alpha,
        )

    def _run_pp(self, arr: np.ndarray) -> TestResult:
        """
        Phillips–Perron test via statsmodels.

        PP is a non-parametric alternative to ADF — it handles
        heteroscedasticity and serial correlation in the error term
        without requiring lag selection.
        """
        from arch.unitroot import PhillipsPerron

        pp = PhillipsPerron(arr, trend=self.adf_regression)
        return TestResult(
            test_name="PP",
            statistic=float(pp.stat),
            p_value=float(pp.pvalue),
            critical_values={k: float(v) for k, v in pp.critical_values.items()},
            null_hypothesis="Unit root exists (non-stationary)",
            reject_null=pp.pvalue < self.alpha,
        )

    def _combine_verdicts(
        self, adf: TestResult, kpss_res: TestResult
    ) -> StationarityVerdict:
        """
        Four-quadrant decision matrix:

                          KPSS fail to reject    KPSS reject
        ADF reject        STATIONARY             TREND_STATIONARY
        ADF fail to rej   INCONCLUSIVE           NON_STATIONARY
        """
        adf_rejects = adf.reject_null       # rejects unit root → evidence of stationarity
        kpss_rejects = kpss_res.reject_null  # rejects stationarity → evidence of non-stationarity

        if adf_rejects and not kpss_rejects:
            return StationarityVerdict.STATIONARY
        if not adf_rejects and kpss_rejects:
            return StationarityVerdict.NON_STATIONARY
        if adf_rejects and kpss_rejects:
            return StationarityVerdict.TREND_STATIONARY
        return StationarityVerdict.INCONCLUSIVE

    def __repr__(self) -> str:
        fitted = self._result is not None
        return (
            f"StationarityTester("
            f"alpha={self.alpha}, fitted={fitted})"
        )
