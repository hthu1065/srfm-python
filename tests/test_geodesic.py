"""Tests for srfm.geodesic — GeodesicSignal."""

import pytest
import numpy as np

from srfm.geodesic import GeodesicSignal


class TestGeodesicSignalCompute:
    def setup_method(self):
        self.gs = GeodesicSignal()

    def test_output_length_equals_input_length(self):
        gammas = [1.0, 1.05, 1.02, 1.08, 1.1]
        betas = [0.0, 0.2, 0.1, 0.3, 0.35]
        result = self.gs.compute(gammas, betas)
        assert len(result) == len(gammas)

    def test_first_element_is_zero(self):
        gammas = [1.0, 1.1, 1.2]
        betas = [0.1, 0.2, 0.3]
        result = self.gs.compute(gammas, betas)
        assert result[0] == pytest.approx(0.0)

    def test_empty_input_returns_empty(self):
        result = self.gs.compute([], [])
        assert result == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            self.gs.compute([1.0, 1.1], [0.1])

    def test_positive_beta_positive_increasing_gamma_positive_signal(self):
        gammas = [1.0, 1.1]
        betas = [0.3, 0.3]
        result = self.gs.compute(gammas, betas)
        assert result[1] > 0

    def test_positive_beta_decreasing_gamma_negative_signal(self):
        gammas = [1.1, 1.0]
        betas = [0.3, 0.3]
        result = self.gs.compute(gammas, betas)
        assert result[1] < 0

    def test_negative_beta_increasing_gamma_negative_signal(self):
        gammas = [1.0, 1.1]
        betas = [-0.3, -0.3]
        result = self.gs.compute(gammas, betas)
        assert result[1] < 0

    def test_zero_beta_gives_zero_signal(self):
        gammas = [1.0, 1.1, 1.2]
        betas = [0.0, 0.0, 0.0]
        result = self.gs.compute(gammas, betas)
        # sign(0) = 0, so all signals should be 0.
        assert all(s == pytest.approx(0.0) for s in result)

    def test_known_value(self):
        gammas = [1.0, 2.0]
        betas = [0.0, 1.0]  # sign(1.0) = 1
        result = self.gs.compute(gammas, betas)
        # (2.0 - 1.0) / 2.0 * 1 = 0.5
        assert result[1] == pytest.approx(0.5)

    def test_single_element(self):
        result = self.gs.compute([1.5], [0.3])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0)


class TestGeodesicDeviation:
    def setup_method(self):
        self.gs = GeodesicSignal()

    def test_output_length_equals_input_length(self):
        signal = np.array([0.0, 0.01, -0.02, 0.005, 0.03, -0.01, 0.02, -0.005])
        result = self.gs.deviation(signal, window=3)
        assert len(result) == len(signal)

    def test_window_less_than_2_raises(self):
        signal = np.ones(10)
        with pytest.raises(ValueError, match="window"):
            self.gs.deviation(signal, window=1)

    def test_constant_signal_returns_zero(self):
        signal = np.ones(20) * 5.0
        result = self.gs.deviation(signal, window=5)
        # After window fills, std=0 → result is 0.
        finite = result[~np.isnan(result)]
        assert np.allclose(finite, 0.0, atol=1e-10)

    def test_nan_for_insufficient_window(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        result = self.gs.deviation(signal, window=5)
        # With window=5, first bar (only 1 point) should be NaN.
        assert np.isnan(result[0])

    def test_zscore_last_element_is_finite(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(0, 0.01, 50)
        result = self.gs.deviation(signal, window=20)
        assert np.isfinite(result[-1])


class TestGeodesicComputeArray:
    def setup_method(self):
        self.gs = GeodesicSignal()

    def test_output_length_matches_input(self):
        gammas = np.array([1.0, 1.1, 1.2, 1.05, 1.15])
        betas = np.array([0.0, 0.2, 0.3, 0.1, 0.25])
        result = self.gs.compute_array(gammas, betas)
        assert len(result) == len(gammas)

    def test_first_element_is_zero(self):
        gammas = np.ones(5) * 1.5
        betas = np.ones(5) * 0.3
        result = self.gs.compute_array(gammas, betas)
        assert result[0] == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            self.gs.compute_array(np.ones(3), np.ones(4))

    def test_all_finite(self):
        rng = np.random.default_rng(42)
        gammas = 1.0 + rng.uniform(0, 0.5, 50)
        betas = rng.uniform(-0.9, 0.9, 50)
        result = self.gs.compute_array(gammas, betas)
        assert np.all(np.isfinite(result))

    def test_consistency_with_list_method(self):
        gammas = [1.0, 1.05, 1.02, 1.08]
        betas = [0.0, 0.2, 0.1, 0.3]
        list_result = self.gs.compute(gammas, betas)
        arr_result = self.gs.compute_array(np.array(gammas), np.array(betas))
        np.testing.assert_allclose(arr_result, list_result, rtol=1e-12)
