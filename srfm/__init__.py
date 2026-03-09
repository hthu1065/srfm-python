"""
srfm-python — Python SDK for Special Relativity Financial Modeling.

Provides pure-Python implementations of the SRFM C++ engine algorithms:
  - BetaVelocity, LorentzFactor, RelativisticSignal (core types)
  - SRFMEngine (full OHLCV pipeline)
  - SpacetimeManifold (metric tensor, Christoffel symbols, spacetime interval)
  - GeodesicSignal (geodesic deviation signal)
  - Pandas accessor: df.srfm.beta(), .gamma(), .geodesic(), .run()
  - Polars plugin:   SRFMPolars(df).beta(), .gamma(), .geodesic(), .run()
"""

from .core import (
    BETA_MAX,
    C_MARKET,
    BetaVelocity,
    LorentzFactor,
    RelativisticSignal,
    compute_beta_array,
    compute_gamma_array,
)
from .engine import SRFMEngine
from .geodesic import GeodesicSignal
from .manifold import SpacetimeManifold

# Register pandas accessor when pandas is available.
try:
    from . import pandas_ext as _pandas_ext  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
__all__ = [
    "BETA_MAX",
    "C_MARKET",
    "BetaVelocity",
    "LorentzFactor",
    "RelativisticSignal",
    "SRFMEngine",
    "GeodesicSignal",
    "SpacetimeManifold",
    "compute_beta_array",
    "compute_gamma_array",
]
