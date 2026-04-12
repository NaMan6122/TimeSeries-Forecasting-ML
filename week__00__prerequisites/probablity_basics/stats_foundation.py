from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

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

