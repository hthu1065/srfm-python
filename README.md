# srfm-python

Python SDK for **Special Relativity Financial Modeling** (SRFM) — pure-Python implementation of the Lorentz factor pipeline for OHLCV market data.

## Install

```bash
pip install -e ".[dev]"          # development install with test deps
pip install -e ".[dev,polars]"   # include polars support
```

## Quickstart

```python
import pandas as pd
import srfm  # registers df.srfm accessor

df = pd.read_csv("ohlcv.csv")   # needs open, high, low, close, volume

# Individual signals
beta_series  = df.srfm.beta()          # pd.Series of β values
gamma_series = df.srfm.gamma()         # pd.Series of γ values (all ≥ 1)
geodesic     = df.srfm.geodesic(window=20)
interval     = df.srfm.spacetime_interval()  # scalar float

# Full pipeline
enriched = df.srfm.run()
print(enriched[["beta", "gamma", "relativistic_return", "geodesic_signal"]].tail())
```

## Polars

```python
import polars as pl
from srfm.polars_ext import SRFMPolars

pl_df = pl.read_csv("ohlcv.csv")
srfm_pl = SRFMPolars(pl_df)
enriched = srfm_pl.run()
```

## Core types

| Type | Description |
|---|---|
| `BetaVelocity` | β = v/c, clamped to `(-0.9999, 0.9999)` |
| `LorentzFactor` | γ = 1/√(1-β²), always ≥ 1 |
| `RelativisticSignal` | γ · m_eff · raw_value |
| `SpacetimeManifold` | 4×4 metric tensor, Christoffel symbols |
| `GeodesicSignal` | Rolling geodesic deviation proxy |
| `SRFMEngine` | Full OHLCV → signal pipeline |

## Pipeline

1. OHLCV → log returns: `r = ln(close_t / close_{t-1})`
2. Returns → β: `β = r / (max_velocity · rolling_max|r|)`, clamped to `BETA_MAX`
3. β → γ: `γ = 1/√(1-β²)`
4. Relativistic return: `γ · m_eff · r`
5. Spacetime interval: `ds² = -(c·dt)² + dr₁² + dr₂² + dr₃²`
6. Geodesic signal: `(Δγ/γ) · sign(β)`

## Tests

```bash
python -m pytest tests/ -v
```
