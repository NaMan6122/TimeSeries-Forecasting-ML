from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

class DescriptiveStats:

    def __init__(self, data: npt.ArrayLike) -> None:
        self._data = np.asarray(data, dtype=np.float64)

    # --- univariate properties --- #

    @property
    def mean(self) -> float | npt.NDArray[np.float64]:
        return np.mean(self._data, axis=0)

    @property
    def variance(self) -> float | npt.NDArray[np.float64]:
        """Population variance (ddof=0). Use .sample_variance for ddof=1."""
        return np.var(self._data, axis=0, ddof=0)

    @property
    def sample_variance(self) -> float | npt.NDArray[np.float64]:
        """Sample variance (ddof=1) — unbiased estimator of σ²."""
        return np.var(self._data, axis=0, ddof=1)

    @property
    def std(self) -> float | npt.NDArray[np.float64]:
        return np.std(self._data, axis=0, ddof=1)

    # --- multivariate methods --- #

    def covariance_matrix(self, ddof: int = 1) -> npt.NDArray[np.float64]:
        """
        Covariance matrix of the columns.

        Covariance(X, Y) = E[(X - μ_X)(Y - μ_Y)]
        Positive → X and Y tend to move together.
        Negative → they move in opposite directions.
        """
        if self._data.ndim == 1:
            return np.array([[np.var(self._data, ddof=ddof)]])
        return np.cov(self._data.T, ddof=ddof)

    def correlation_matrix(self) -> npt.NDArray[np.float64]:
        """
        Pearson correlation matrix.

        Correlation is covariance scaled to [-1, 1] so you can compare
        relationships across variables with different scales.
        """
        if self._data.ndim == 1:
            return np.array([[1.0]])
        return np.corrcoef(self._data.T)

    def __repr__(self) -> str:
        shape = self._data.shape
        return f"DescriptiveStats(shape={shape}, mean={np.round(self.mean, 4)})"

# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

class HypothesisTest:
    """
    Wrappers for common hypothesis tests.

    These thin wrappers exist so:
      (a) the test result is a typed object (TestResult), not a raw tuple
      (b) the null hypothesis is stated explicitly alongside the numbers
      (c) tests are easily extended or mocked in unit tests

    Parameters
    ----------
    alpha : float
        Significance level. Default 0.05.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def one_sample_t_test(
        self,
        sample: npt.ArrayLike,
        popmean: float = 0.0,
    ) -> TestResult:
        """
        Test H₀: the sample mean equals popmean.

        Used for checking whether OLS residuals have zero mean —
        a key diagnostic for model correctness.
        """
        arr = np.asarray(sample, dtype=np.float64)
        stat, p = stats.ttest_1samp(arr, popmean=popmean)
        return TestResult(
            test_name="One-sample t-test",
            statistic=float(stat),
            p_value=float(p),
            reject_null=p < self.alpha,
            alpha=self.alpha,
            null_hypothesis=f"mean == {popmean}",
        )

    def two_sample_t_test(
        self,
        a: npt.ArrayLike,
        b: npt.ArrayLike,
        equal_var: bool = False,
    ) -> TestResult:
        """
        Test H₀: the means of two independent samples are equal.

        equal_var=False uses Welch's t-test (safer default — does not
        assume equal population variances).
        """
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        stat, p = stats.ttest_ind(a_arr, b_arr, equal_var=equal_var)
        variant = "Student" if equal_var else "Welch"
        return TestResult(
            test_name=f"Two-sample t-test ({variant})",
            statistic=float(stat),
            p_value=float(p),
            reject_null=p < self.alpha,
            alpha=self.alpha,
            null_hypothesis="mean(a) == mean(b)",
        )

    def __repr__(self) -> str:
        return f"HypothesisTest(alpha={self.alpha})"
    
@dataclass
class TestResult:
    """Holds the outcome of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    reject_null: bool #if p_value is less than alpha, null hypo is rejected.
    alpha: float = 0.05
    null_hypothesis: str = ""

    def __repr__(self) -> str:
        decision = "REJECT H₀" if self.reject_null else "FAIL TO REJECT H₀"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} → {decision} (α={self.alpha})"
        )

