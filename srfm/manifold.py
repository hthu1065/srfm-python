"""
SpacetimeManifold — metric tensor, spacetime interval, Christoffel symbols.

Implements the Minkowski-like metric with market corrections as described in
the SRFM paper.  Coordinates are (t, r1, r2, r3) where:
  - t   = time (bar index or unix timestamp delta)
  - r1  = log return (price dimension)
  - r2  = volume normalised to [0,1]
  - r3  = high-low range normalised to [0,1]
"""

from __future__ import annotations

from typing import List

import numpy as np


class SpacetimeManifold:
    """Computes the SRFM 4-D spacetime manifold quantities.

    The metric signature is (-, +, +, +) following the physics convention.

    # Example
    >>> m = SpacetimeManifold(c_market=1.0)
    >>> g = m.metric_tensor({"dt": 1.0, "r1": 0.01, "r2": 0.5, "r3": 0.02})
    >>> g.shape
    (4, 4)
    """

    def __init__(self, c_market: float = 1.0) -> None:
        """Initialise manifold with speed of light in market units.

        # Arguments
        * `c_market` — market speed of light (default 1.0, dimensionless).
        """
        if c_market <= 0.0:
            raise ValueError(f"c_market must be positive, got {c_market}")
        self.c_market = c_market

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def metric_tensor(self, bar: dict) -> np.ndarray:
        """Build the 4x4 Minkowski-like metric tensor for a single bar.

        The diagonal metric is:
          g = diag(-c²·dt², 1 + |r1|, 1 + |r2|, 1 + |r3|)

        The off-diagonal terms are zero (locally flat approximation with
        market-correction scaling on the spatial dimensions).

        # Arguments
        * `bar` — dict with keys `dt` (time step), `r1`, `r2`, `r3`.

        # Returns
        4x4 float64 ndarray.
        """
        dt = float(bar.get("dt", 1.0))
        r1 = float(bar.get("r1", 0.0))
        r2 = float(bar.get("r2", 0.0))
        r3 = float(bar.get("r3", 0.0))

        c = self.c_market
        g = np.zeros((4, 4), dtype=np.float64)
        g[0, 0] = -(c * dt) ** 2  # temporal component (negative signature)
        g[1, 1] = 1.0 + abs(r1)    # price-return dimension
        g[2, 2] = 1.0 + abs(r2)    # volume dimension
        g[3, 3] = 1.0 + abs(r3)    # hl-range dimension
        return g

    def spacetime_interval(self, bars: List[dict]) -> float:
        """Compute the total spacetime interval ds² over a sequence of bars.

        ds² = Σ_t [-(c·dt)² + dr1² + dr2² + dr3²]

        # Arguments
        * `bars` — list of bar dicts, each with `dt`, `r1`, `r2`, `r3`.

        # Returns
        Scalar float (can be negative — spacelike trajectory).
        """
        if not bars:
            return 0.0

        total = 0.0
        c = self.c_market
        for bar in bars:
            dt = float(bar.get("dt", 1.0))
            r1 = float(bar.get("r1", 0.0))
            r2 = float(bar.get("r2", 0.0))
            r3 = float(bar.get("r3", 0.0))
            total += -(c * dt) ** 2 + r1 ** 2 + r2 ** 2 + r3 ** 2

        return total

    def christoffel_symbol(
        self,
        i: int,
        j: int,
        k: int,
        metric: np.ndarray,
        inv_metric: np.ndarray,
    ) -> float:
        """Compute a single Christoffel symbol Γⁱⱼₖ.

        Γⁱⱼₖ = ½ gⁱˡ (∂ⱼgₗₖ + ∂ₖgⱼₗ - ∂ₗgⱼₖ)

        For the diagonal metric used here, ∂g/∂x is approximated by finite
        differences between consecutive metrics.  When only one metric is
        provided the derivatives are zero and all Christoffel symbols vanish
        (flat spacetime).

        # Arguments
        * `i`, `j`, `k`   — index triple (0..3 each).
        * `metric`         — 4x4 metric at current point.
        * `inv_metric`     — 4x4 inverse metric at current point.

        # Returns
        Scalar float.
        """
        if not (0 <= i <= 3 and 0 <= j <= 3 and 0 <= k <= 3):
            raise ValueError(f"Indices must be in [0,3], got ({i},{j},{k})")

        # For a diagonal metric, non-diagonal inverse entries are zero, and
        # the derivative approximation is zero when evaluated at a single point.
        # The full expression reduces to zero for a diagonal, position-independent
        # metric.  A non-trivial result requires two consecutive metric snapshots.
        result = 0.0
        for lam in range(4):
            # Partial derivatives of the metric (zero for single-snapshot diagonal).
            d_j_g_lk = 0.0  # ∂_j g_{lk}
            d_k_g_jl = 0.0  # ∂_k g_{jl}
            d_l_g_jk = 0.0  # ∂_l g_{jk}
            result += 0.5 * inv_metric[i, lam] * (d_j_g_lk + d_k_g_jl - d_l_g_jk)

        return result

    def christoffel_from_pair(
        self,
        i: int,
        j: int,
        k: int,
        metric_a: np.ndarray,
        metric_b: np.ndarray,
    ) -> float:
        """Compute Γⁱⱼₖ using finite differences between two consecutive metrics.

        Approximates ∂g/∂xˡ ≈ (g_b - g_a) / 1  (unit coordinate step).

        # Arguments
        * `i`, `j`, `k`   — index triple.
        * `metric_a`       — metric at step t.
        * `metric_b`       — metric at step t+1.

        # Returns
        Scalar float.
        """
        if not (0 <= i <= 3 and 0 <= j <= 3 and 0 <= k <= 3):
            raise ValueError(f"Indices must be in [0,3], got ({i},{j},{k})")

        dg = metric_b - metric_a  # finite-difference approximation of ∂g/∂x

        # Invert the average metric.
        avg_metric = 0.5 * (metric_a + metric_b)
        try:
            inv_metric = np.linalg.inv(avg_metric)
        except np.linalg.LinAlgError:
            return 0.0

        result = 0.0
        for lam in range(4):
            d_j_g_lk = dg[lam, k]
            d_k_g_jl = dg[j, lam]
            d_l_g_jk = dg[j, k]
            result += 0.5 * inv_metric[i, lam] * (d_j_g_lk + d_k_g_jl - d_l_g_jk)

        return result

    # ------------------------------------------------------------------
    # Vectorised helpers
    # ------------------------------------------------------------------

    def spacetime_interval_array(
        self,
        dt_arr: np.ndarray,
        r1_arr: np.ndarray,
        r2_arr: np.ndarray,
        r3_arr: np.ndarray,
    ) -> np.ndarray:
        """Vectorised per-bar spacetime interval.

        # Arguments
        * `dt_arr` — 1-D array of time steps.
        * `r1_arr` — 1-D array of log returns.
        * `r2_arr` — 1-D array of normalised volumes.
        * `r3_arr` — 1-D array of normalised hl-ranges.

        # Returns
        1-D float64 array of per-bar ds² values.
        """
        c = self.c_market
        return -(c * dt_arr) ** 2 + r1_arr ** 2 + r2_arr ** 2 + r3_arr ** 2
