from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

# ------------------------------------------------------------------
# Implementation
# ------------------------------------------------------------------

@dataclass
class OLSResult:
    """Stores the outcome of a single OLS fit. Immutable by intent."""

    coefficients: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    r_squared: float
    sse: float # sum of squared errors / residuals -> OLS minimises this.
    sst: float # total sum of squares

    def __repr__(self) -> str:
        coef_str = ", ".join(f"{c:.6f}" for c in self.coefficients)
        return (
            f"OLSResult(coefficients=[{coef_str}], "
            f"R²={self.r_squared:.4f}, SSE={self.sse:.4f})"
        )

class OLSRegression:
    """
    Ordinary Least Squares regression via the normal equations.

    The normal equation β = (XᵀX)⁻¹Xᵀy gives the exact closed-form
    solution because SSR (sum of squared residuals) is a convex quadratic
    in β — it has exactly one global minimum, so calculus gives us the
    answer directly without any iterative optimisation.

    Parameters
    ----------
    fit_intercept : bool
        If True, prepend a column of ones to X so β₀ is estimated.
        Default is True.

    use_pseudoinverse : bool
        If True, use Moore-Penrose pseudoinverse (np.linalg.pinv) instead
        of the standard inverse. Recommended when X is near-singular or
        when features are highly correlated. Default is True.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 1))
    >>> y = 3.0 + 2.5 * X.ravel() + rng.standard_normal(100) * 0.5
    >>> model = OLSRegression()
    >>> result = model.fit(X, y)
    >>> result.coefficients  # should be close to [3.0, 2.5]
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        use_pseudoinverse: bool = True,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.use_pseudoinverse = use_pseudoinverse
        self._result: Optional[OLSResult] = None

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
    ) -> OLSResult:
        """
        Fit OLS on (X, y) and store the result.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        OLSResult
        """
        X_arr, y_arr = self._validate_inputs(X, y)
        X_design = self._build_design_matrix(X_arr)

        # β = (XᵀX)⁻¹ Xᵀy, Using pinv handles the near-singular case gracefully by returning
        # the minimum-norm solution when XᵀX is not invertible.
        if self.use_pseudoinverse:
            beta = np.linalg.pinv(X_design) @ y_arr
        else:
            beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ y_arr)

        y_hat = X_design @ beta
        residuals = y_arr - y_hat

        # SSE = Σ(yᵢ - ŷᵢ)²
        sse = float(np.sum(residuals ** 2))

        # SST = Σ(yᵢ - ȳ)²
        sst = float(np.sum((y_arr - y_arr.mean()) ** 2))

        r_squared = 1.0 - sse / sst if sst > 0 else 0.0

        self._result = OLSResult(
            coefficients=beta,
            residuals=residuals,
            r_squared=r_squared,
            sse=sse,
            sst=sst,
        )
        return self._result

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Predict y for new X values using the fitted coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self._result is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() before .predict().")
        X_arr, _ = self._validate_inputs(X, np.zeros(1))
        X_design = self._build_design_matrix(X_arr)
        return X_design @ self._result.coefficients

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """β vector. Index 0 is intercept if fit_intercept=True."""
        self._assert_fitted()
        return self._result.coefficients

    @property
    def intercept(self) -> float:
        """β₀ (intercept). Only meaningful if fit_intercept=True."""
        self._assert_fitted()
        return float(self._result.coefficients[0])

    @property
    def slopes(self) -> npt.NDArray[np.float64]:
        """β₁..βₚ (slope terms), excluding intercept."""
        self._assert_fitted()
        start = 1 if self.fit_intercept else 0
        return self._result.coefficients[start:]

    @property
    def r_squared(self) -> float:
        self._assert_fitted()
        return self._result.r_squared

    @property
    def residuals(self) -> npt.NDArray[np.float64]:
        self._assert_fitted()
        return self._result.residuals

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def residual_summary(self) -> dict[str, float]:
        """
        Return key residual statistics.

        A well-specified OLS model should have:
          - mean ≈ 0 (guaranteed if fit_intercept=True)
          - skewness ≈ 0  (symmetric distribution)
          - excess kurtosis ≈ 0 (normal-ish tails)
        """
        self._assert_fitted()
        from scipy import stats

        e = self._result.residuals
        skew, _ = stats.skewtest(e)
        kurt, _ = stats.kurtosistest(e)
        _, shapiro_p = stats.shapiro(e)

        return {
            "mean": float(np.mean(e)),
            "std": float(np.std(e)),
            "skew_statistic": float(skew),
            "kurtosis_statistic": float(kurt),
            "shapiro_wilk_p": float(shapiro_p),
            "normality_ok": shapiro_p > 0.05,
        }
    
    # ------------------------------------------------------------------
    #Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self._result is not None
        return (
            f"OLSRegression("
            f"fit_intercept={self.fit_intercept}, "
            f"use_pseudoinverse={self.use_pseudoinverse}, "
            f"fitted={fitted})"
        )

    @staticmethod
    def _validate_inputs(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {X_arr.shape}")
        return X_arr, y_arr

    def _build_design_matrix(
        self,
        X: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Prepend a column of ones if fit_intercept=True."""
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=np.float64)
            return np.hstack([ones, X])
        return X

    def _assert_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Model is not fitted yet. Call .fit(X, y) first.")