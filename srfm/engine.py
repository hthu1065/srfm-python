"""
SRFMEngine — main pipeline orchestrating the full SRFM signal computation.

Pipeline stages:
  1. OHLCV bars → log returns
  2. Returns → BetaVelocity (β = r / (max_velocity · rolling_max|r|), clamped)
  3. β → LorentzFactor (γ = 1/√(1-β²))
  4. γ + momentum → RelativisticSignal (adjusted = γ · m_eff · raw)
  5. Spacetime interval: ds² = -(c·dt)² + dr1² + dr2² + dr3²
  6. Geodesic deviation signal

## Responsibility
Owns the stateless batch pipeline over a pandas DataFrame.

## Guarantees
- Deterministic: same input always produces same output.
- Non-panicking: all edge cases (single bar, zero returns) handled gracefully.
- Column-agnostic: column names are matched case-insensitively.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .core import (
    BETA_MAX,
    BetaVelocity,
    LorentzFactor,
    RelativisticSignal,
    compute_beta_array,
    compute_gamma_array,
)
from .geodesic import GeodesicSignal
from .manifold import SpacetimeManifold


class SRFMEngine:
    """Main SRFM pipeline engine.

    # Arguments
    * `max_velocity`   — denominator scale for beta computation (default 1.0).
    * `effective_mass` — m_eff in the relativistic signal formula (default 1.0).
    * `c_market`       — speed of light in market units (default 1.0).
    * `rolling_window` — lookback for max absolute return normalisation (default 20).

    # Example
    >>> import pandas as pd
    >>> engine = SRFMEngine()
    >>> df = pd.DataFrame({
    ...     "open":   [100, 101, 102, 103, 104],
    ...     "high":   [101, 102, 103, 104, 105],
    ...     "low":    [ 99, 100, 101, 102, 103],
    ...     "close":  [100.5, 101.5, 102.5, 103.5, 104.5],
    ...     "volume": [1000, 1100, 900, 1200, 1050],
    ... })
    >>> result = engine.run(df)
    >>> "gamma" in result.columns
    True
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(
        self,
        max_velocity: float = 1.0,
        effective_mass: float = 1.0,
        c_market: float = 1.0,
        rolling_window: int = 20,
    ) -> None:
        if max_velocity <= 0.0:
            raise ValueError(f"max_velocity must be positive, got {max_velocity}")
        if c_market <= 0.0:
            raise ValueError(f"c_market must be positive, got {c_market}")
        if rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {rolling_window}")

        self.max_velocity = max_velocity
        self.effective_mass = effective_mass
        self.c_market = c_market
        self.rolling_window = rolling_window

        self._manifold = SpacetimeManifold(c_market=c_market)
        self._geodesic = GeodesicSignal()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full SRFM pipeline on an OHLCV DataFrame.

        Adds columns to a copy of the input DataFrame:
        - `log_return`          — ln(close_t / close_{t-1}), NaN at t=0
        - `beta`                — market velocity β
        - `gamma`               — Lorentz factor γ
        - `relativistic_return` — γ · m_eff · log_return
        - `volume_norm`         — volume / rolling_max(volume)
        - `hl_range_norm`       — (high - low) / close, normalised
        - `spacetime_interval`  — per-bar ds²
        - `geodesic_signal`     — raw geodesic deviation
        - `geodesic_deviation`  — rolling z-score of geodesic_signal

        # Arguments
        * `df` — DataFrame with columns open, high, low, close, volume
                 (case-insensitive).

        # Returns
        New DataFrame with all original columns plus the above additions.

        # Raises
        `ValueError` if required columns are missing.
        """
        df = self._normalise_columns(df)
        self._validate_columns(df)

        out = df.copy()
        n = len(out)

        close = out["close"].to_numpy(dtype=np.float64)
        high = out["high"].to_numpy(dtype=np.float64)
        low = out["low"].to_numpy(dtype=np.float64)
        volume = out["volume"].to_numpy(dtype=np.float64)

        # 1. Log returns.
        log_returns = np.full(n, np.nan, dtype=np.float64)
        if n > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns[1:] = np.log(close[1:] / close[:-1])

        out["log_return"] = log_returns

        # 2. Beta.
        betas = compute_beta_array(log_returns, self.max_velocity, self.rolling_window)
        out["beta"] = betas

        # 3. Gamma.
        gammas = compute_gamma_array(betas)
        out["gamma"] = gammas

        # 4. Relativistic return.
        filled_returns = np.where(np.isnan(log_returns), 0.0, log_returns)
        out["relativistic_return"] = gammas * self.effective_mass * filled_returns

        # 5. Normalised volume and hl-range for spacetime coordinates.
        vol_max = np.maximum.accumulate(np.where(np.isnan(volume), 0.0, volume))
        vol_max = np.where(vol_max < 1e-12, 1.0, vol_max)
        volume_norm = volume / vol_max

        hl_range = high - low
        with np.errstate(divide="ignore", invalid="ignore"):
            hl_range_norm = np.where(close > 1e-12, hl_range / close, 0.0)

        out["volume_norm"] = volume_norm
        out["hl_range_norm"] = hl_range_norm

        # 6. Spacetime interval (dt = 1 bar by convention).
        dt_arr = np.ones(n, dtype=np.float64)
        out["spacetime_interval"] = self._manifold.spacetime_interval_array(
            dt_arr, filled_returns, volume_norm, hl_range_norm
        )

        # 7. Geodesic signal and rolling deviation.
        geodesic_raw = self._geodesic.compute_array(gammas, betas)
        out["geodesic_signal"] = geodesic_raw

        window = min(self.rolling_window, max(2, n))
        out["geodesic_deviation"] = self._geodesic.deviation(geodesic_raw, window=window)

        return out

    def run_bar(self, ohlcv: dict, prev_close: Optional[float] = None) -> dict:
        """Run the pipeline for a single OHLCV bar.

        # Arguments
        * `ohlcv`       — dict with keys open, high, low, close, volume.
        * `prev_close`  — previous bar's closing price (for return calculation).
                          If None, log_return is 0.0 and beta is 0.0.

        # Returns
        Dict with original keys plus: log_return, beta, gamma, relativistic_return,
        spacetime_interval.
        """
        close = float(ohlcv.get("close", 0.0))
        high = float(ohlcv.get("high", close))
        low = float(ohlcv.get("low", close))
        volume = float(ohlcv.get("volume", 0.0))

        if prev_close is not None and prev_close > 1e-12:
            log_return = np.log(close / prev_close)
        else:
            log_return = 0.0

        beta = BetaVelocity(log_return / (self.max_velocity + 1e-12))
        gamma = LorentzFactor.from_beta(beta)

        sig = RelativisticSignal.compute(
            raw_value=log_return,
            beta=beta,
            effective_mass=self.effective_mass,
        )

        hl_range_norm = (high - low) / close if close > 1e-12 else 0.0

        bar_dict = dict(ohlcv)
        bar_dict["log_return"] = log_return
        bar_dict["beta"] = beta.value
        bar_dict["gamma"] = gamma.value
        bar_dict["relativistic_return"] = sig.adjusted_value
        bar_dict["spacetime_interval"] = (
            -(self.c_market * 1.0) ** 2 + log_return ** 2 + volume ** 2 + hl_range_norm ** 2
        )
        return bar_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with column names lowercased."""
        return df.rename(columns={c: c.lower() for c in df.columns})

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )

    def signals_from_bars(self, bars: List[dict]) -> List[RelativisticSignal]:
        """Compute a RelativisticSignal per bar from a list of OHLCV dicts.

        # Arguments
        * `bars` — list of OHLCV dicts in chronological order.

        # Returns
        List of RelativisticSignal, same length as input.
        """
        signals: List[RelativisticSignal] = []
        prev_close: Optional[float] = None
        for bar in bars:
            close = float(bar.get("close", 0.0))
            if prev_close is not None and prev_close > 1e-12:
                log_return = float(np.log(close / prev_close))
            else:
                log_return = 0.0
            beta = BetaVelocity(log_return / (self.max_velocity + 1e-12))
            sig = RelativisticSignal.compute(
                raw_value=log_return,
                beta=beta,
                effective_mass=self.effective_mass,
                timestamp=bar.get("timestamp"),
            )
            signals.append(sig)
            prev_close = close
        return signals
