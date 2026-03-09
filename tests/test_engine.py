"""Tests for srfm.engine — SRFMEngine pipeline."""

import math
import pytest
import numpy as np
import pandas as pd

from srfm.engine import SRFMEngine
from srfm.core import BETA_MAX


def make_ohlcv(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.uniform(500, 2000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestSRFMEngineInit:
    def test_default_construction(self):
        engine = SRFMEngine()
        assert engine.max_velocity == 1.0
        assert engine.effective_mass == 1.0
        assert engine.c_market == 1.0
        assert engine.rolling_window == 20

    def test_invalid_max_velocity_raises(self):
        with pytest.raises(ValueError, match="max_velocity"):
            SRFMEngine(max_velocity=0.0)

    def test_invalid_c_market_raises(self):
        with pytest.raises(ValueError, match="c_market"):
            SRFMEngine(c_market=-1.0)

    def test_invalid_rolling_window_raises(self):
        with pytest.raises(ValueError, match="rolling_window"):
            SRFMEngine(rolling_window=0)


class TestSRFMEngineRun:
    def setup_method(self):
        self.engine = SRFMEngine()
        self.df = make_ohlcv(50)

    def test_run_returns_dataframe(self):
        result = self.engine.run(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_run_adds_required_columns(self):
        result = self.engine.run(self.df)
        required = {
            "log_return", "beta", "gamma", "relativistic_return",
            "spacetime_interval", "geodesic_signal", "geodesic_deviation",
        }
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        result = self.engine.run(self.df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_output_length_equals_input_length(self):
        result = self.engine.run(self.df)
        assert len(result) == len(self.df)

    def test_gamma_always_ge_1(self):
        result = self.engine.run(self.df)
        assert (result["gamma"] >= 1.0).all(), "gamma must be >= 1.0 everywhere"

    def test_beta_within_bounds(self):
        result = self.engine.run(self.df)
        assert (result["beta"].abs() <= BETA_MAX + 1e-10).all()

    def test_spacetime_interval_is_finite(self):
        result = self.engine.run(self.df)
        assert result["spacetime_interval"].notna().all()
        assert np.isfinite(result["spacetime_interval"].to_numpy()).all()

    def test_geodesic_signal_same_length(self):
        result = self.engine.run(self.df)
        assert len(result["geodesic_signal"]) == len(self.df)

    def test_log_return_nan_at_first_bar(self):
        result = self.engine.run(self.df)
        assert math.isnan(result["log_return"].iloc[0])

    def test_log_return_nonzero_after_first_bar(self):
        result = self.engine.run(self.df)
        # At least some returns should be non-zero for random data.
        assert result["log_return"].iloc[1:].abs().sum() > 0

    def test_case_insensitive_columns(self):
        df_upper = self.df.rename(columns=str.upper)
        result = self.engine.run(df_upper)
        assert "gamma" in result.columns

    def test_missing_column_raises(self):
        df_missing = self.df.drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            self.engine.run(df_missing)

    def test_run_does_not_mutate_input(self):
        original_cols = list(self.df.columns)
        _ = self.engine.run(self.df)
        assert list(self.df.columns) == original_cols

    def test_relativistic_return_amplified(self):
        # For non-zero returns, relativistic_return should differ from log_return.
        result = self.engine.run(self.df)
        # At least for bars where gamma > 1 and return != 0, amplification applies.
        diff = (result["relativistic_return"] - result["log_return"]).abs().sum()
        assert diff > 0


class TestSRFMEngineEdgeCases:
    def test_single_bar(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1000.0],
        })
        result = SRFMEngine().run(df)
        assert len(result) == 1
        assert result["gamma"].iloc[0] >= 1.0

    def test_two_bars(self):
        df = pd.DataFrame({
            "open": [100.0, 101.0], "high": [101.0, 102.0], "low": [99.0, 100.0],
            "close": [100.5, 101.5], "volume": [1000.0, 1100.0],
        })
        result = SRFMEngine().run(df)
        assert len(result) == 2
        assert (result["gamma"] >= 1.0).all()

    def test_all_zero_returns(self):
        # All bars with identical close — log returns are 0.
        df = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        })
        result = SRFMEngine().run(df)
        # gamma should be 1.0 everywhere (zero beta).
        assert np.allclose(result["gamma"].to_numpy(), 1.0)

    def test_very_high_volatility(self):
        # Extreme price swings — beta should be clamped.
        rng = np.random.default_rng(99)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.5, 30))
        df = pd.DataFrame({
            "open": close,
            "high": close * 1.1,
            "low": close * 0.9,
            "close": close,
            "volume": np.ones(30) * 1000,
        })
        result = SRFMEngine().run(df)
        assert (result["gamma"] >= 1.0).all()
        assert (result["beta"].abs() <= BETA_MAX + 1e-10).all()

    def test_custom_effective_mass(self):
        df = make_ohlcv(20)
        engine1 = SRFMEngine(effective_mass=1.0)
        engine2 = SRFMEngine(effective_mass=2.0)
        r1 = engine1.run(df)
        r2 = engine2.run(df)
        # relativistic_return should be doubled.
        np.testing.assert_allclose(
            r2["relativistic_return"].to_numpy(),
            r1["relativistic_return"].to_numpy() * 2.0,
            rtol=1e-9,
        )


class TestRunBar:
    def test_run_bar_returns_dict(self):
        engine = SRFMEngine()
        result = engine.run_bar({"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000})
        assert isinstance(result, dict)

    def test_run_bar_has_required_keys(self):
        engine = SRFMEngine()
        result = engine.run_bar({"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000})
        for key in ["beta", "gamma", "relativistic_return", "spacetime_interval"]:
            assert key in result

    def test_run_bar_with_prev_close(self):
        engine = SRFMEngine()
        result = engine.run_bar(
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
            prev_close=100.0,
        )
        expected_return = math.log(101 / 100)
        assert result["log_return"] == pytest.approx(expected_return, rel=1e-9)

    def test_run_bar_gamma_ge_1(self):
        engine = SRFMEngine()
        result = engine.run_bar({"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000})
        assert result["gamma"] >= 1.0
