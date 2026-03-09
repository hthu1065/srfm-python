"""
GeodesicSignal — geodesic deviation signal computation.

The geodesic signal is a proxy for the curvature of the market trajectory
in SRFM spacetime.  It measures how much the Lorentz factor deviates from
a smooth geodesic path, scaled by the direction of velocity (sign of beta).
"""

from __future__ import annotations

from typing import List

import numpy as np


class GeodesicSignal:
    """Computes geodesic deviation signals from gamma and beta series.

    # Example
    >>> gs = GeodesicSignal()
    >>> betas  = [0.1, 0.2, 0.15, 0.3]
    >>> gammas = [1.005, 1.02, 1.011, 1.046]
    >>> signals = gs.compute(gammas, betas)
    >>> len(signals) == len(betas)
    True
    """

    def compute(
        self,
        gammas: List[float],
        betas: List[float],
    ) -> List[float]:
        """Compute per-bar geodesic deviation signal.

        Signal_t = (Δγ_t / γ_t) · sign(β_t)

        where Δγ_t = γ_t - γ_{t-1}.  At t=0 the signal is 0.0.

        # Arguments
        * `gammas` — list of Lorentz factors (all >= 1.0).
        * `betas`  — list of beta values, same length as gammas.

        # Returns
        List of float, same length as inputs.
        """
        if len(gammas) != len(betas):
            raise ValueError(
                f"gammas and betas must have the same length, "
                f"got {len(gammas)} and {len(betas)}"
            )
        if not gammas:
            return []

        g = np.asarray(gammas, dtype=np.float64)
        b = np.asarray(betas, dtype=np.float64)

        signals = np.zeros(len(g), dtype=np.float64)
        for t in range(1, len(g)):
            delta_gamma = g[t] - g[t - 1]
            # Avoid division by zero for gamma == 0 (should never happen, but defensive).
            gamma_denom = g[t] if g[t] > 1e-12 else 1e-12
            signals[t] = (delta_gamma / gamma_denom) * np.sign(b[t])

        return signals.tolist()

    def deviation(
        self,
        signal_series: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Rolling geodesic deviation (z-score of the raw geodesic signal).

        deviation_t = (signal_t - rolling_mean_t) / rolling_std_t

        NaN is returned for bars where the rolling window has insufficient data
        (fewer than 2 observations).

        # Arguments
        * `signal_series` — 1-D array of raw geodesic signals.
        * `window`        — rolling window size (default 20).

        # Returns
        1-D float64 ndarray of same length as `signal_series`.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")

        s = np.asarray(signal_series, dtype=np.float64)
        n = len(s)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            sub = s[start : i + 1]
            if len(sub) < 2:
                continue
            mu = sub.mean()
            sigma = sub.std(ddof=1)
            if sigma < 1e-12:
                result[i] = 0.0
            else:
                result[i] = (s[i] - mu) / sigma

        return result

    def compute_array(
        self,
        gammas: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version of compute(), returns numpy array.

        # Arguments
        * `gammas` — 1-D float64 array.
        * `betas`  — 1-D float64 array, same length.

        # Returns
        1-D float64 ndarray.
        """
        if len(gammas) != len(betas):
            raise ValueError(
                f"gammas and betas must have the same length, "
                f"got {len(gammas)} and {len(betas)}"
            )

        g = np.asarray(gammas, dtype=np.float64)
        b = np.asarray(betas, dtype=np.float64)

        signals = np.zeros(len(g), dtype=np.float64)
        if len(g) < 2:
            return signals

        delta_gamma = np.diff(g, prepend=g[0])
        gamma_denom = np.where(g > 1e-12, g, 1e-12)
        signals = (delta_gamma / gamma_denom) * np.sign(b)
        signals[0] = 0.0  # first bar has no previous, by convention
        return signals
