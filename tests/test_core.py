"""Tests for srfm.core — BetaVelocity, LorentzFactor, RelativisticSignal."""

import math
import pytest
import numpy as np

from srfm.core import (
    BETA_MAX,
    BetaVelocity,
    LorentzFactor,
    RelativisticSignal,
    compute_beta_array,
    compute_gamma_array,
)


# ---------------------------------------------------------------------------
# BetaVelocity
# ---------------------------------------------------------------------------

class TestBetaVelocity:
    def test_normal_value_stored_unchanged(self):
        bv = BetaVelocity(0.5)
        assert bv.value == pytest.approx(0.5)

    def test_zero_value(self):
        bv = BetaVelocity(0.0)
        assert bv.value == 0.0

    def test_positive_clamping_at_beta_max(self):
        bv = BetaVelocity(2.0)
        assert bv.value == pytest.approx(BETA_MAX)

    def test_negative_clamping_at_minus_beta_max(self):
        bv = BetaVelocity(-5.0)
        assert bv.value == pytest.approx(-BETA_MAX)

    def test_exactly_beta_max_passes_through(self):
        bv = BetaVelocity(BETA_MAX)
        assert bv.value == pytest.approx(BETA_MAX)

    def test_value_just_below_beta_max_not_clamped(self):
        v = BETA_MAX - 1e-6
        bv = BetaVelocity(v)
        assert bv.value == pytest.approx(v)

    def test_float_conversion(self):
        bv = BetaVelocity(0.3)
        assert float(bv) == pytest.approx(0.3)

    def test_repr_contains_value(self):
        bv = BetaVelocity(0.42)
        assert "0.42" in repr(bv)

    def test_equality(self):
        assert BetaVelocity(0.5) == BetaVelocity(0.5)
        assert BetaVelocity(0.5) != BetaVelocity(0.6)

    def test_equality_with_non_beta_returns_not_implemented(self):
        bv = BetaVelocity(0.5)
        assert bv.__eq__(0.5) is NotImplemented

    def test_beta_max_constant_is_0_9999(self):
        assert BETA_MAX == 0.9999


# ---------------------------------------------------------------------------
# LorentzFactor
# ---------------------------------------------------------------------------

class TestLorentzFactor:
    def test_beta_zero_gives_gamma_one(self):
        lf = LorentzFactor.from_beta(BetaVelocity(0.0))
        assert lf.value == pytest.approx(1.0)

    def test_known_beta_gives_correct_gamma(self):
        # β = 0.6 → γ = 1/√(1-0.36) = 1/√0.64 = 1/0.8 = 1.25
        lf = LorentzFactor.from_beta(BetaVelocity(0.6))
        assert lf.value == pytest.approx(1.25, rel=1e-9)

    def test_gamma_always_ge_1(self):
        for v in [0.0, 0.1, 0.5, 0.9, 0.999]:
            lf = LorentzFactor.from_beta(BetaVelocity(v))
            assert lf.value >= 1.0

    def test_negative_beta_same_gamma_as_positive(self):
        lf_pos = LorentzFactor.from_beta(BetaVelocity(0.5))
        lf_neg = LorentzFactor.from_beta(BetaVelocity(-0.5))
        assert lf_pos.value == pytest.approx(lf_neg.value)

    def test_beta_near_max_gives_large_gamma(self):
        lf = LorentzFactor.from_beta(BetaVelocity(BETA_MAX))
        assert lf.value > 50.0

    def test_invalid_gamma_less_than_1_raises(self):
        with pytest.raises(ValueError, match="LorentzFactor must be >= 1.0"):
            LorentzFactor(0.5)

    def test_infinite_gamma_raises(self):
        with pytest.raises(ValueError, match="finite"):
            LorentzFactor(float("inf"))

    def test_nan_gamma_raises(self):
        with pytest.raises(ValueError, match="finite"):
            LorentzFactor(float("nan"))

    def test_float_conversion(self):
        lf = LorentzFactor(1.5)
        assert float(lf) == pytest.approx(1.5)

    def test_beta_0_5_gamma_approx(self):
        # β=0.5 → γ = 1/√0.75 ≈ 1.1547
        lf = LorentzFactor.from_beta(BetaVelocity(0.5))
        assert lf.value == pytest.approx(1.0 / math.sqrt(0.75), rel=1e-9)


# ---------------------------------------------------------------------------
# RelativisticSignal
# ---------------------------------------------------------------------------

class TestRelativisticSignal:
    def test_adjusted_value_equals_gamma_times_mass_times_raw(self):
        beta = BetaVelocity(0.6)
        gamma = LorentzFactor.from_beta(beta)
        sig = RelativisticSignal.compute(0.01, beta, effective_mass=2.0)
        expected = gamma.value * 2.0 * 0.01
        assert sig.adjusted_value == pytest.approx(expected)

    def test_default_mass_is_one(self):
        beta = BetaVelocity(0.5)
        gamma = LorentzFactor.from_beta(beta)
        sig = RelativisticSignal.compute(0.05, beta)
        assert sig.adjusted_value == pytest.approx(gamma.value * 0.05)

    def test_zero_raw_value_gives_zero_adjusted(self):
        beta = BetaVelocity(0.9)
        sig = RelativisticSignal.compute(0.0, beta)
        assert sig.adjusted_value == pytest.approx(0.0)

    def test_timestamp_stored(self):
        beta = BetaVelocity(0.1)
        sig = RelativisticSignal.compute(0.01, beta, timestamp=1234567890.0)
        assert sig.timestamp == pytest.approx(1234567890.0)

    def test_gamma_matches_computed_from_beta(self):
        beta = BetaVelocity(0.3)
        sig = RelativisticSignal.compute(0.02, beta)
        expected_gamma = LorentzFactor.from_beta(beta)
        assert sig.gamma.value == pytest.approx(expected_gamma.value)

    def test_raw_value_stored(self):
        beta = BetaVelocity(0.2)
        sig = RelativisticSignal.compute(0.007, beta)
        assert sig.raw_value == pytest.approx(0.007)


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------

class TestComputeBetaArray:
    def test_output_length_matches_input(self):
        returns = np.array([np.nan, 0.01, -0.02, 0.005, 0.03])
        betas = compute_beta_array(returns)
        assert len(betas) == len(returns)

    def test_all_values_within_beta_max(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.05, 200)
        betas = compute_beta_array(returns)
        assert np.all(np.abs(betas) <= BETA_MAX + 1e-10)

    def test_zero_returns_give_zero_beta(self):
        returns = np.zeros(10)
        betas = compute_beta_array(returns)
        assert np.allclose(betas, 0.0)

    def test_single_element_array(self):
        returns = np.array([0.0])
        betas = compute_beta_array(returns)
        assert len(betas) == 1

    def test_nan_return_treated_as_zero(self):
        returns = np.array([np.nan, 0.01, 0.02])
        betas = compute_beta_array(returns)
        assert np.isfinite(betas[0])


class TestComputeGammaArray:
    def test_zero_beta_gives_gamma_one(self):
        betas = np.zeros(5)
        gammas = compute_gamma_array(betas)
        assert np.allclose(gammas, 1.0)

    def test_all_gammas_ge_1(self):
        betas = np.linspace(-0.9, 0.9, 100)
        gammas = compute_gamma_array(betas)
        assert np.all(gammas >= 1.0)

    def test_known_beta_value(self):
        betas = np.array([0.6])
        gammas = compute_gamma_array(betas)
        assert gammas[0] == pytest.approx(1.25, rel=1e-9)

    def test_output_length_matches_input(self):
        betas = np.linspace(-0.5, 0.5, 50)
        gammas = compute_gamma_array(betas)
        assert len(gammas) == 50
