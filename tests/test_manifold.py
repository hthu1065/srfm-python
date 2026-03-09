"""Tests for srfm.manifold — SpacetimeManifold."""

import pytest
import numpy as np

from srfm.manifold import SpacetimeManifold


def make_bar(**kwargs):
    defaults = {"dt": 1.0, "r1": 0.01, "r2": 0.5, "r3": 0.02}
    defaults.update(kwargs)
    return defaults


class TestSpacetimeManifoldInit:
    def test_default_c_market(self):
        m = SpacetimeManifold()
        assert m.c_market == 1.0

    def test_custom_c_market(self):
        m = SpacetimeManifold(c_market=2.5)
        assert m.c_market == 2.5

    def test_invalid_c_market_raises(self):
        with pytest.raises(ValueError, match="c_market"):
            SpacetimeManifold(c_market=0.0)

    def test_negative_c_market_raises(self):
        with pytest.raises(ValueError, match="c_market"):
            SpacetimeManifold(c_market=-1.0)


class TestMetricTensor:
    def setup_method(self):
        self.m = SpacetimeManifold()

    def test_shape_is_4x4(self):
        g = self.m.metric_tensor(make_bar())
        assert g.shape == (4, 4)

    def test_temporal_component_is_negative(self):
        g = self.m.metric_tensor(make_bar(dt=1.0))
        assert g[0, 0] < 0

    def test_spatial_components_are_positive(self):
        g = self.m.metric_tensor(make_bar())
        assert g[1, 1] > 0
        assert g[2, 2] > 0
        assert g[3, 3] > 0

    def test_off_diagonal_are_zero(self):
        g = self.m.metric_tensor(make_bar())
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert g[i, j] == pytest.approx(0.0)

    def test_metric_depends_on_c_market(self):
        m1 = SpacetimeManifold(c_market=1.0)
        m2 = SpacetimeManifold(c_market=2.0)
        g1 = m1.metric_tensor(make_bar(dt=1.0))
        g2 = m2.metric_tensor(make_bar(dt=1.0))
        assert g2[0, 0] == pytest.approx(4.0 * g1[0, 0])

    def test_zero_r1_gives_spatial_component_1(self):
        g = self.m.metric_tensor(make_bar(r1=0.0, r2=0.0, r3=0.0))
        assert g[1, 1] == pytest.approx(1.0)
        assert g[2, 2] == pytest.approx(1.0)
        assert g[3, 3] == pytest.approx(1.0)

    def test_missing_keys_use_defaults(self):
        g = self.m.metric_tensor({})
        assert g.shape == (4, 4)
        assert g[0, 0] < 0


class TestSpacetimeInterval:
    def setup_method(self):
        self.m = SpacetimeManifold(c_market=1.0)

    def test_empty_bars_returns_zero(self):
        assert self.m.spacetime_interval([]) == pytest.approx(0.0)

    def test_single_bar_with_zero_displacement(self):
        bar = make_bar(dt=1.0, r1=0.0, r2=0.0, r3=0.0)
        result = self.m.spacetime_interval([bar])
        # ds² = -(1*1)² + 0 + 0 + 0 = -1.0
        assert result == pytest.approx(-1.0)

    def test_spacelike_trajectory_is_positive(self):
        # Large spatial displacements dominate.
        bar = make_bar(dt=0.001, r1=5.0, r2=5.0, r3=5.0)
        result = self.m.spacetime_interval([bar])
        assert result > 0

    def test_timelike_trajectory_is_negative(self):
        # Large time step, small spatial displacements.
        bar = make_bar(dt=100.0, r1=0.0001, r2=0.0001, r3=0.0001)
        result = self.m.spacetime_interval([bar])
        assert result < 0

    def test_multiple_bars_sums_correctly(self):
        bars = [make_bar(dt=1.0, r1=0.0, r2=0.0, r3=0.0)] * 3
        result = self.m.spacetime_interval(bars)
        assert result == pytest.approx(-3.0)

    def test_interval_is_real(self):
        bars = [make_bar() for _ in range(10)]
        result = self.m.spacetime_interval(bars)
        assert np.isfinite(result)


class TestChristoffelSymbol:
    def setup_method(self):
        self.m = SpacetimeManifold()
        bar = make_bar()
        self.g = self.m.metric_tensor(bar)
        self.g_inv = np.diag(1.0 / np.diag(self.g))  # diagonal inverse

    def test_flat_spacetime_all_zero(self):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result = self.m.christoffel_symbol(i, j, k, self.g, self.g_inv)
                    assert result == pytest.approx(0.0)

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError, match="Indices must be"):
            self.m.christoffel_symbol(4, 0, 0, self.g, self.g_inv)


class TestChristoffelFromPair:
    def setup_method(self):
        self.m = SpacetimeManifold()

    def test_identical_metrics_gives_zero(self):
        bar = make_bar()
        g = self.m.metric_tensor(bar)
        result = self.m.christoffel_from_pair(0, 0, 0, g, g)
        assert result == pytest.approx(0.0)

    def test_returns_finite_value_for_different_metrics(self):
        g_a = self.m.metric_tensor(make_bar(r1=0.01))
        g_b = self.m.metric_tensor(make_bar(r1=0.05))
        result = self.m.christoffel_from_pair(1, 1, 1, g_a, g_b)
        assert np.isfinite(result)


class TestSpacetimeIntervalArray:
    def test_output_length_matches_input(self):
        m = SpacetimeManifold()
        n = 20
        dt = np.ones(n)
        r1 = np.random.default_rng(1).normal(0, 0.01, n)
        r2 = np.ones(n) * 0.5
        r3 = np.ones(n) * 0.02
        result = m.spacetime_interval_array(dt, r1, r2, r3)
        assert len(result) == n

    def test_all_finite(self):
        m = SpacetimeManifold()
        n = 10
        dt = np.ones(n)
        r1 = np.zeros(n)
        r2 = np.zeros(n)
        r3 = np.zeros(n)
        result = m.spacetime_interval_array(dt, r1, r2, r3)
        assert np.all(np.isfinite(result))
