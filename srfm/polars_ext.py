"""
Polars integration for SRFM.

Provides a plugin class `SRFMPolars` that mirrors the pandas accessor API
for polars DataFrames.  Uses a wrapper approach (not a native Polars extension)
for maximum compatibility.

Usage::

    from srfm.polars_ext import SRFMPolars

    pl_df = pl.DataFrame({...})
    srfm_pl = SRFMPolars(pl_df)
    beta_series = srfm_pl.beta()
    enriched_df = srfm_pl.run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

from .core import compute_beta_array, compute_gamma_array
from .geodesic import GeodesicSignal
from .manifold import SpacetimeManifold

if TYPE_CHECKING:
    pass


def _require_polars() -> None:
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install it with: pip install srfm-python[polars]"
        )


class SRFMPolars:
    """Polars plugin class providing SRFM computations on polars DataFrames.

    Mirrors the pandas accessor API so users can switch backends with minimal
    code changes.

    # Arguments
    * `df` — a polars DataFrame with columns: open, high, low, close, volume
             (case-insensitive).

    # Example
    >>> import polars as pl
    >>> from srfm.polars_ext import SRFMPolars
    >>> df = pl.DataFrame({
    ...     "open":   [100.0, 101.0, 102.0],
    ...     "high":   [101.0, 102.0, 103.0],
    ...     "low":    [ 99.0, 100.0, 101.0],
    ...     "close":  [100.5, 101.5, 102.5],
    ...     "volume": [1000.0, 1100.0, 900.0],
    ... })
    >>> srfm = SRFMPolars(df)
    >>> beta = srfm.beta()
    >>> len(beta) == len(df)
    True
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, df: "pl.DataFrame") -> None:
        _require_polars()
        self._df = self._normalise(df)
        self._check_columns()

    # ------------------------------------------------------------------
    # Individual signal accessors
    # ------------------------------------------------------------------

    def beta(
        self,
        max_velocity: float = 1.0,
        rolling_window: int = 20,
    ) -> "pl.Series":
        """Compute per-bar market velocity beta.

        # Returns
        polars Series of Float64.
        """
        _require_polars()
        close = self._df["close"].to_numpy().astype(np.float64)
        n = len(close)
        log_returns = np.full(n, np.nan, dtype=np.float64)
        if n > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns[1:] = np.log(close[1:] / close[:-1])
        betas = compute_beta_array(log_returns, max_velocity, rolling_window)
        return pl.Series("beta", betas, dtype=pl.Float64)

    def gamma(
        self,
        max_velocity: float = 1.0,
        rolling_window: int = 20,
    ) -> "pl.Series":
        """Compute per-bar Lorentz factor gamma.

        # Returns
        polars Series of Float64, all values >= 1.0.
        """
        _require_polars()
        betas = self.beta(max_velocity=max_velocity, rolling_window=rolling_window).to_numpy()
        gammas = compute_gamma_array(betas)
        return pl.Series("gamma", gammas, dtype=pl.Float64)

    def geodesic(
        self,
        window: int = 20,
        max_velocity: float = 1.0,
        rolling_window: int = 20,
    ) -> "pl.Series":
        """Compute rolling geodesic deviation signal.

        # Returns
        polars Series of Float64.
        """
        _require_polars()
        betas = self.beta(max_velocity=max_velocity, rolling_window=rolling_window).to_numpy()
        gammas = compute_gamma_array(betas)
        gs = GeodesicSignal()
        raw = gs.compute_array(gammas, betas)
        effective_window = min(window, max(2, len(raw)))
        dev = gs.deviation(raw, window=effective_window)
        return pl.Series("geodesic_deviation", dev, dtype=pl.Float64)

    def spacetime_interval(self, c_market: float = 1.0) -> float:
        """Compute total spacetime interval over the entire DataFrame.

        # Returns
        Scalar float.
        """
        _require_polars()
        close = self._df["close"].to_numpy().astype(np.float64)
        high = self._df["high"].to_numpy().astype(np.float64)
        low = self._df["low"].to_numpy().astype(np.float64)
        volume = self._df["volume"].to_numpy().astype(np.float64)

        n = len(close)
        log_returns = np.zeros(n, dtype=np.float64)
        if n > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns[1:] = np.log(close[1:] / close[:-1])

        vol_max = np.maximum.accumulate(np.where(np.isnan(volume), 0.0, volume))
        vol_max = np.where(vol_max < 1e-12, 1.0, vol_max)
        volume_norm = volume / vol_max
        hl_range_norm = np.where(close > 1e-12, (high - low) / close, 0.0)
        dt_arr = np.ones(n, dtype=np.float64)

        manifold = SpacetimeManifold(c_market=c_market)
        intervals = manifold.spacetime_interval_array(dt_arr, log_returns, volume_norm, hl_range_norm)
        return float(np.nansum(intervals))

    def run(
        self,
        max_velocity: float = 1.0,
        effective_mass: float = 1.0,
        c_market: float = 1.0,
        rolling_window: int = 20,
    ) -> "pl.DataFrame":
        """Run the full SRFM pipeline and return an enriched polars DataFrame.

        Adds columns: log_return, beta, gamma, relativistic_return,
        volume_norm, hl_range_norm, spacetime_interval, geodesic_signal,
        geodesic_deviation.

        # Returns
        New polars DataFrame with SRFM columns added.
        """
        _require_polars()
        from .engine import SRFMEngine
        import pandas as pd

        # Convert to pandas, run engine, convert back.
        pandas_df = self._df.to_pandas()
        engine = SRFMEngine(
            max_velocity=max_velocity,
            effective_mass=effective_mass,
            c_market=c_market,
            rolling_window=rolling_window,
        )
        result_pd = engine.run(pandas_df)
        return pl.from_pandas(result_pd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(df: "pl.DataFrame") -> "pl.DataFrame":
        """Return df with column names lowercased."""
        return df.rename({c: c.lower() for c in df.columns})

    def _check_columns(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self._df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required SRFM columns: {sorted(missing)}"
            )
