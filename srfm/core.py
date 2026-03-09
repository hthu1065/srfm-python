"""
Core SRFM types: BetaVelocity, LorentzFactor, RelativisticSignal.

These mirror the C++ structs in the SRFM engine and implement the exact
same algorithms in pure Python + numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Maximum safe beta — matches C++ BETA_MAX_SAFE constant.
BETA_MAX: float = 0.9999

# Speed of light in market units (dimensionless by convention).
C_MARKET: float = 1.0


@dataclass
class BetaVelocity:
    """Market velocity β = v/c, clamped to the open interval (-BETA_MAX, BETA_MAX).

    A value of 0 means no directional momentum; ±BETA_MAX approaches the
    relativistic speed limit and produces very large Lorentz factors.

    # Arguments
    * `value` — raw velocity ratio before clamping is applied.

    # Returns
    A `BetaVelocity` whose `.value` is in (-BETA_MAX, BETA_MAX).

    # Panics
    This function never panics.

    # Example
    >>> bv = BetaVelocity(0.5)
    >>> bv.value
    0.5
    >>> BetaVelocity(2.0).value == BETA_MAX
    True
    """

    _value: float = field(repr=False, default=0.0)
    BETA_MAX: float = field(default=BETA_MAX, init=False, repr=False, compare=False)

    def __init__(self, value: float) -> None:
        self._value = float(np.clip(value, -BETA_MAX, BETA_MAX))

    @property
    def value(self) -> float:
        """Clamped velocity value in (-BETA_MAX, BETA_MAX)."""
        return self._value

    def __repr__(self) -> str:
        return f"BetaVelocity(value={self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BetaVelocity):
            return self._value == other._value
        return NotImplemented

    def __float__(self) -> float:
        return self._value


@dataclass
class LorentzFactor:
    """Lorentz factor γ = 1/√(1-β²), always ≥ 1.0.

    At β=0, γ=1 (no time dilation / momentum amplification).
    As β → BETA_MAX, γ grows large, amplifying the relativistic signal.

    # Arguments
    * `value` — pre-computed Lorentz factor; must be ≥ 1.0.

    # Example
    >>> lf = LorentzFactor.from_beta(BetaVelocity(0.0))
    >>> lf.value
    1.0
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 1.0:
            raise ValueError(
                f"LorentzFactor must be >= 1.0 and finite, got {self.value}"
            )

    @classmethod
    def from_beta(cls, beta: BetaVelocity) -> "LorentzFactor":
        """Compute γ from a BetaVelocity instance.

        # Arguments
        * `beta` — market velocity; value must be in (-BETA_MAX, BETA_MAX).

        # Returns
        `LorentzFactor` with value = 1/√(1-β²).

        # Example
        >>> lf = LorentzFactor.from_beta(BetaVelocity(0.5))
        >>> abs(lf.value - 1.1547005383792515) < 1e-12
        True
        """
        b = beta.value
        denominator = math.sqrt(1.0 - b * b)
        return cls(1.0 / denominator)

    def __float__(self) -> float:
        return self.value


@dataclass
class RelativisticSignal:
    """A market signal amplified by the Lorentz factor.

    adjusted_value = γ · effective_mass · raw_value

    # Arguments
    * `raw_value`      — un-amplified signal (e.g. log return).
    * `gamma`          — Lorentz factor for this bar.
    * `adjusted_value` — γ · m_eff · raw_value.
    * `timestamp`      — optional unix timestamp for the bar.

    # Example
    >>> sig = RelativisticSignal(0.01, LorentzFactor(1.5), 0.015)
    >>> sig.adjusted_value
    0.015
    """

    raw_value: float
    gamma: LorentzFactor
    adjusted_value: float
    timestamp: Optional[float] = None

    @classmethod
    def compute(
        cls,
        raw_value: float,
        beta: BetaVelocity,
        effective_mass: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> "RelativisticSignal":
        """Construct a RelativisticSignal from raw inputs.

        # Arguments
        * `raw_value`      — un-amplified signal.
        * `beta`           — market velocity.
        * `effective_mass` — mass-like scaling parameter (default 1.0).
        * `timestamp`      — optional unix timestamp.

        # Returns
        `RelativisticSignal` with adjusted_value = γ · m_eff · raw_value.
        """
        gamma = LorentzFactor.from_beta(beta)
        adjusted = gamma.value * effective_mass * raw_value
        return cls(
            raw_value=raw_value,
            gamma=gamma,
            adjusted_value=adjusted,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Vectorised numpy helpers — used by engine.py
# ---------------------------------------------------------------------------


def compute_beta_array(
    returns: np.ndarray,
    max_velocity: float = 1.0,
    rolling_window: int = 20,
) -> np.ndarray:
    """Compute an array of beta values from log returns.

    β_t = r_t / (max_velocity · rolling_max_|r|_t), clamped to BETA_MAX.

    For the first `rolling_window - 1` bars, the expanding window is used
    so that beta is defined from bar 0 onward.

    # Arguments
    * `returns`        — 1-D array of log returns (NaN at index 0 is handled).
    * `max_velocity`   — scaling denominator (default 1.0).
    * `rolling_window` — lookback for max absolute return (default 20).

    # Returns
    1-D float64 array with same length as `returns`, values in [-BETA_MAX, BETA_MAX].
    """
    r = np.where(np.isnan(returns), 0.0, returns)
    abs_r = np.abs(r)

    n = len(r)
    rolling_max = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - rolling_window + 1)
        window_max = abs_r[start : i + 1].max()
        rolling_max[i] = window_max

    # Avoid division by zero — treat flat markets as zero beta.
    denom = max_velocity * rolling_max
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_beta = np.where(denom > 1e-12, r / denom, 0.0)

    return np.clip(raw_beta, -BETA_MAX, BETA_MAX)


def compute_gamma_array(betas: np.ndarray) -> np.ndarray:
    """Vectorised Lorentz factor computation.

    # Arguments
    * `betas` — 1-D array of beta values in [-BETA_MAX, BETA_MAX].

    # Returns
    1-D float64 array of gamma values, all >= 1.0.
    """
    b2 = betas ** 2
    # Clip to avoid numerical sqrt of negative due to floating-point noise.
    return 1.0 / np.sqrt(np.clip(1.0 - b2, 1e-12, None))
