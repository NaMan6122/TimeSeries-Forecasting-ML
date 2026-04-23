"""
TimeSeriesLoader — Week 1 data ingestion and validation.

Responsibilities:
  - load a time series from a CSV or from statsmodels built-in datasets
  - enforce a DatetimeIndex
  - validate for gaps, duplicate timestamps, and constant series
  - expose the raw series and a clean summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class LoadSummary:
    """Metadata produced when a series is loaded and validated."""

    name: str
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    has_missing: bool
    missing_count: int
    has_duplicates: bool
    is_constant: bool
    freq: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            f"LoadSummary('{self.name}')",
            f"  n_obs       : {self.n_obs}",
            f"  period      : {self.start.date()} → {self.end.date()}",
            f"  frequency   : {self.freq or 'irregular'}",
            f"  missing     : {self.missing_count}",
            f"  duplicates  : {self.has_duplicates}",
            f"  is_constant : {self.is_constant}",
        ]
        if self.warnings:
            lines += [f"  WARNING     : {w}" for w in self.warnings]
        return "\n".join(lines)

class TimeSeriesLoader:
    """
    Load and validate a univariate time series.

    Supported sources
    -----------------
    - CSV file (path or Path object) with a datetime column
    - statsmodels built-in dataset name:
        "airline"  → AirPassengers (monthly)
        "sunspots" → Sunspot counts (annual)
        "co2"      → Mauna Loa CO₂ (weekly)

    Parameters
    ----------
    name : str
        Human-readable name for the series (used in repr and plots).

    """

    def __init__(self, name: str = "series") -> None:
        self.name = name
        self._series: Optional[pd.Series] = None
        self._summary: Optional[LoadSummary] = None

    def from_statsmodels(self, dataset: str) -> tuple[pd.Series, LoadSummary]:
        """
        Load a built-in statsmodels dataset by short name.

        Returns
        -------
        (series, summary)
        """
        series = self._fetch_builtin(dataset)
        return self._finalise(series)

    def from_csv(
        self,
        path: str | Path,
        date_col: str,
        value_col: str,
        date_format: Optional[str] = None,
        freq: Optional[str] = None,
    ) -> tuple[pd.Series, LoadSummary]:
        """
        Load from a CSV file.

        Parameters
        ----------
        path       : file path
        date_col   : column name containing timestamps
        value_col  : column name containing the numeric series
        date_format: strftime format string (optional; pandas auto-detects)
        freq       : force a frequency string, e.g. "MS" (optional)
        """
        df = pd.read_csv(path, parse_dates=[date_col])
        series = df.set_index(date_col)[value_col]
        series.index = pd.DatetimeIndex(series.index)
        if freq:
            series.index.freq = freq
        return self._finalise(series)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def series(self) -> pd.Series:
        self._assert_loaded()
        return self._series

    @property
    def summary(self) -> LoadSummary:
        self._assert_loaded()
        return self._summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_builtin(self, dataset: str) -> pd.Series:
        import statsmodels.api as sm

        dataset_map = {
            "airline": ("datasets.get_rdataset", "AirPassengers", "datasets"),
            "sunspots": ("sunspots", None, None),
            "co2": ("co2", None, None),
        }

        key = dataset.lower()
        if key not in dataset_map:
            raise ValueError(
                f"Unknown built-in dataset '{dataset}'. "
                f"Choose from: {list(dataset_map)}"
            )

        if key == "airline":
            data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
            series = pd.Series(
                data["value"].values,
                index=pd.date_range("1949-01", periods=len(data), freq="MS"),
                name="AirPassengers",
            )
        elif key == "sunspots":
            data = sm.datasets.sunspots.load_pandas().data
            series = pd.Series(
                data["SUNACTIVITY"].values,
                index=pd.to_datetime(data["YEAR"].astype(int).astype(str)),
                name="Sunspots",
            )
        elif key == "co2":
            data = sm.datasets.co2.load_pandas().data
            series = data["co2"].dropna()
            series.name = "CO2"

        return series

    def _finalise(self, series: pd.Series) -> tuple[pd.Series, LoadSummary]:
        """Run validation, build summary, store state."""
        warnings: list[str] = []

        if series.index.freq is None:
            inferred = pd.infer_freq(series.index)
            if inferred:
                series.index.freq = inferred

        missing_count = int(series.isna().sum())
        has_missing = missing_count > 0
        if has_missing:
            warnings.append(f"{missing_count} missing values detected")

        has_duplicates = bool(series.index.duplicated().any())
        if has_duplicates:
            warnings.append("Duplicate timestamps detected — index is not unique")

        is_constant = bool(series.nunique() <= 1)
        if is_constant:
            warnings.append("Series appears constant — check data source")

        summary = LoadSummary(
            name=self.name,
            n_obs=len(series),
            freq=str(series.index.freq) if series.index.freq else None,
            start=series.index.min(),
            end=series.index.max(),
            has_missing=has_missing,
            missing_count=missing_count,
            has_duplicates=has_duplicates,
            is_constant=is_constant,
            warnings=warnings,
        )

        self._series = series.rename(self.name)
        self._summary = summary
        return self._series, self._summary

    def _assert_loaded(self) -> None:
        if self._series is None:
            raise RuntimeError(
                "No series loaded yet. Call from_statsmodels() or from_csv() first."
            )

    def __repr__(self) -> str:
        loaded = self._series is not None
        return f"TimeSeriesLoader(name='{self.name}', loaded={loaded})"
