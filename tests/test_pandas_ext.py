"""Tests for srfm.pandas_ext — SRFMAccessor."""

import pytest
import numpy as np
import pandas as pd

import srfm  # Triggers accessor registration
from srfm.core import BETA_MAX


def make_ohlcv(n: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.uniform(500, 2000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestPandasAccessorRegistration:
    def test_accessor_is_available(self):
        df = make_ohlcv()
        assert hasattr(df, "srfm")

    def test_accessor_is_srfm_accessor_type(self):
        from srfm.pandas_ext import SRFMAccessor
        df = make_ohlcv()
        assert isinstance(df.srfm, SRFMAccessor)


class TestPandasAccessorBeta:
    def test_beta_returns_series(self):
        df = make_ohlcv()
        result = df.srfm.beta()
        assert isinstance(result, pd.Series)

    def test_beta_dtype_is_float64(self):
        df = make_ohlcv()
        result = df.srfm.beta()
        assert result.dtype == np.float64

    def test_beta_length_equals_df_length(self):
        df = make_ohlcv(25)
        result = df.srfm.beta()
        assert len(result) == 25

    def test_beta_within_bounds(self):
        df = make_ohlcv()
        result = df.srfm.beta()
        assert (result.abs() <= BETA_MAX + 1e-10).all()

    def test_beta_index_matches_df_index(self):
        df = make_ohlcv()
        result = df.srfm.beta()
        assert result.index.equals(df.index)

    def test_beta_name_is_beta(self):
        df = make_ohlcv()
        assert df.srfm.beta().name == "beta"


class TestPandasAccessorGamma:
    def test_gamma_returns_series(self):
        df = make_ohlcv()
        result = df.srfm.gamma()
        assert isinstance(result, pd.Series)

    def test_gamma_all_ge_1(self):
        df = make_ohlcv()
        result = df.srfm.gamma()
        assert (result >= 1.0).all()

    def test_gamma_length_equals_df_length(self):
        df = make_ohlcv(15)
        assert len(df.srfm.gamma()) == 15

    def test_gamma_name_is_gamma(self):
        df = make_ohlcv()
        assert df.srfm.gamma().name == "gamma"

    def test_gamma_index_matches_df_index(self):
        df = make_ohlcv()
        result = df.srfm.gamma()
        assert result.index.equals(df.index)


class TestPandasAccessorGeodesic:
    def test_geodesic_returns_series(self):
        df = make_ohlcv()
        result = df.srfm.geodesic()
        assert isinstance(result, pd.Series)

    def test_geodesic_length_equals_df_length(self):
        df = make_ohlcv(40)
        result = df.srfm.geodesic()
        assert len(result) == 40

    def test_geodesic_name(self):
        df = make_ohlcv()
        assert df.srfm.geodesic().name == "geodesic_deviation"

    def test_geodesic_finite_where_not_nan(self):
        df = make_ohlcv(30)
        result = df.srfm.geodesic(window=10)
        finite_vals = result.dropna()
        assert np.all(np.isfinite(finite_vals.to_numpy()))


class TestPandasAccessorSpacetimeInterval:
    def test_spacetime_interval_returns_scalar(self):
        df = make_ohlcv()
        result = df.srfm.spacetime_interval()
        assert isinstance(result, float)

    def test_spacetime_interval_is_finite(self):
        df = make_ohlcv()
        result = df.srfm.spacetime_interval()
        assert np.isfinite(result)

    def test_spacetime_interval_single_bar(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1000.0],
        })
        result = df.srfm.spacetime_interval()
        assert np.isfinite(result)


class TestPandasAccessorRun:
    def test_run_returns_dataframe(self):
        df = make_ohlcv()
        result = df.srfm.run()
        assert isinstance(result, pd.DataFrame)

    def test_run_adds_srfm_columns(self):
        df = make_ohlcv()
        result = df.srfm.run()
        for col in ["beta", "gamma", "log_return", "relativistic_return",
                    "spacetime_interval", "geodesic_signal", "geodesic_deviation"]:
            assert col in result.columns

    def test_run_gamma_all_ge_1(self):
        df = make_ohlcv()
        result = df.srfm.run()
        assert (result["gamma"] >= 1.0).all()

    def test_run_with_uppercase_columns(self):
        df = make_ohlcv().rename(columns=str.upper)
        result = df.srfm.run()
        assert "gamma" in result.columns

    def test_run_missing_column_raises(self):
        df = make_ohlcv().drop(columns=["volume"])
        with pytest.raises((ValueError, AttributeError)):
            df.srfm.run()
