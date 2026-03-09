"""
Pandas DataFrame accessor for SRFM.

Registers the `.srfm` accessor on pandas DataFrames so that users can call:

    df.srfm.beta()
    df.srfm.gamma()
    df.srfm.geodesic(window=20)
    df.srfm.spacetime_interval()
    df.srfm.run()

Expects columns: open, high, low, close, volume (case-insensitive).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .core import compute_beta_array, compute_gamma_array
from .engine import SRFMEngine
from .geodesic import GeodesicSignal


@pd.api.extensions.register_dataframe_accessor("srfm")
class SRFMAccessor:
    """Pandas DataFrame accessor providing SRFM computations.

    # Example
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "open":   [100.0, 101.0, 102.0],
    ...     "high":   [101.0, 102.0, 103.0],
    ...     "low":    [ 99.0, 100.0, 101.0],
    ...     "close":  [100.5, 101.5, 102.5],
    ...     "volume": [1000.0, 1100.0, 900.0],
    ... })
    >>> df.srfm.beta().dtype
    dtype('float64')
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._df = pandas_obj
        self._engine = SRFMEngine()

    # ------------------------------------------------------------------
    # Individual signal accessors
    # ------------------------------------------------------------------

    def beta(
        self,
        max_velocity: float = 1.0,
        rolling_window: int = 20,
    ) -> pd.Series:
        """Compute per-bar market velocity beta.

        # Arguments
        * `max_velocity`   — velocity scaling denominator (default 1.0).
        * `rolling_window` — lookback for max absolute return (default 20).

        # Returns
        pd.Series of float64 with same index as df.
        """
        df = self._normalise()
        self._check_columns(df)
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(close)
        log_returns = np.full(n, np.nan, dtype=np.float64)
        if n > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns[1:] = np.log(close[1:] / close[:-1])
        betas = compute_beta_array(log_returns, max_velocity, rolling_window)
        return pd.Series(betas, index=self._df.index, name="beta", dtype=np.float64)

    def gamma(
        self,
        max_velocity: float = 1.0,
        rolling_window: int = 20,
    ) -> pd.Series:
        """Compute per-bar Lorentz factor gamma.

        # Arguments
        * `max_velocity`   — velocity scaling denominator (default 1.0).
        * `rolling_window` — lookback for max absolute return (default 20).

        # Returns
        pd.Series of float64, all values >= 1.0.
        """
        betas = self.beta(max_velocity=max_velocity, rolling_window=rolling_window)
        gammas = compute_gamma_array(betas.to_numpy())
        return pd.Series(gammas, index=self._df.index, name="gamma", dtype=np.float64)

    def geodesic(
        self,
        window: int = 20,
        max_velocity: float = 1.0,
        rolling_window: int = 20,
    ) -> pd.Series:
        """Compute rolling geodesic deviation signal.

        # Arguments
        * `window`         — rolling window for z-score (default 20).
        * `max_velocity`   — velocity scaling denominator (default 1.0).
        * `rolling_window` — lookback for beta normalisation (default 20).

        # Returns
        pd.Series of float64.
        """
        betas = self.beta(max_velocity=max_velocity, rolling_window=rolling_window).to_numpy()
        gammas = compute_gamma_array(betas)
        gs = GeodesicSignal()
        raw = gs.compute_array(gammas, betas)
        effective_window = min(window, max(2, len(raw)))
        dev = gs.deviation(raw, window=effective_window)
        return pd.Series(dev, index=self._df.index, name="geodesic_deviation", dtype=np.float64)

    def spacetime_interval(
        self,
        c_market: float = 1.0,
    ) -> float:
        """Compute total spacetime interval over the entire DataFrame.

        ds² = Σ_t [-(c·dt)² + dr1² + dr2² + dr3²]

        # Arguments
        * `c_market` — market speed of light (default 1.0).

        # Returns
        Scalar float.
        """
        from .manifold import SpacetimeManifold

        df = self._normalise()
        self._check_columns(df)

        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

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
    ) -> pd.DataFrame:
        """Run the full SRFM pipeline and return an enriched DataFrame.

        # Arguments
        * `max_velocity`   — velocity scaling denominator (default 1.0).
        * `effective_mass` — m_eff for relativistic signal (default 1.0).
        * `c_market`       — market speed of light (default 1.0).
        * `rolling_window` — lookback window (default 20).

        # Returns
        New DataFrame with all SRFM columns added.
        """
        engine = SRFMEngine(
            max_velocity=max_velocity,
            effective_mass=effective_mass,
            c_market=c_market,
            rolling_window=rolling_window,
        )
        return engine.run(self._df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(self) -> pd.DataFrame:
        return self._df.rename(columns={c: c.lower() for c in self._df.columns})

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = SRFMEngine.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise AttributeError(
                f"DataFrame is missing required SRFM columns: {sorted(missing)}"
            )
