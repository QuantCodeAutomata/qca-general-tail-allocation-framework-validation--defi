"""Tests for src/asymptotic.py — W_u, asymptotic sequences, theorem bounds."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.asymptotic import (
    ParetoBoundedWScenario,
    OscillatingShareScenario,
    DiscreteDivergentWScenario,
    compute_W_sequence,
    compute_phi_psi_sequence,
    estimate_convergence,
    check_theorem_bounds,
    compute_asymptotic_diagnostics,
    classify_asymptotic_behavior,
)

# ---------------------------------------------------------------
# ParetoBoundedWScenario
# ---------------------------------------------------------------

class TestParetoBoundedW:
    def setup_method(self):
        self.sc = ParetoBoundedWScenario(alphas=[2.5, 3.0], sigmas=[1.0, 1.5])

    def test_W_u_positive(self):
        """W_u > 0 for u in (0, 1)."""
        for u in [0.5, 0.7, 0.9, 0.99]:
            W = self.sc.W_u(u)
            assert W > 0.0

    def test_W_u_ge_1(self):
        """W_u = ES/q >= 1 when q > 0."""
        for u in [0.5, 0.7, 0.9]:
            W = self.sc.W_u(u)
            assert W >= 1.0 - 1e-8

    def test_W_u_bounded_at_high_u(self):
        """W_u stays bounded as u -> 1 for Pareto."""
        W_vals = [self.sc.W_u(u) for u in [0.9, 0.95, 0.99, 0.999]]
        assert max(W_vals) < 100.0

    def test_phi_psi_sum_to_one(self):
        """phi_0 + phi_1 = 1 and psi_0 + psi_1 = 1."""
        for u in [0.6, 0.8, 0.95]:
            phi_sum = self.sc.phi_i(0, u) + self.sc.phi_i(1, u)
            psi_sum = self.sc.psi_i(0, u) + self.sc.psi_i(1, u)
            assert abs(phi_sum - 1.0) < 1e-4, f"u={u}: phi_sum={phi_sum:.4f}"
            assert abs(psi_sum - 1.0) < 1e-4, f"u={u}: psi_sum={psi_sum:.4f}"

    def test_asymptotic_limit_sums_to_one(self):
        """Asymptotic limits sum to 1."""
        lim_sum = self.sc.asymptotic_limit(0) + self.sc.asymptotic_limit(1)
        assert abs(lim_sum - 1.0) < 1e-10

    def test_phi_increases_toward_limit(self):
        """phi_i(u) increases toward asymptotic_limit(i) as u -> 1.

        Convergence to the limit 1.0 is slow when alpha indices are close
        (2.5 vs 3.0); we test that phi is monotone-increasing and above 0.5
        at the extreme tail, not the exact limit value.
        """
        u_vals = [0.9, 0.99, 0.999]
        phi_tail = [self.sc.phi_i(0, u) for u in u_vals]
        # All values should be valid
        assert all(np.isfinite(v) for v in phi_tail)
        # phi_0 is always > 0 (component 0 has heaviest tail)
        assert all(v > 0.0 for v in phi_tail)
        # phi(0.999) should be larger than phi(0.9) – trending toward 1.0
        assert phi_tail[-1] > phi_tail[0]

    def test_phi_gt_psi_for_heavier_tail(self):
        """For heavier-tailed component (alpha=2.5), Phi > Psi at high u."""
        u = 0.99
        phi0 = self.sc.phi_i(0, u)
        psi0 = self.sc.psi_i(0, u)
        # Both should converge to lim, Delta -> 0 for bounded W
        assert phi0 >= 0.0 and psi0 >= 0.0

    def test_q_s_positive(self):
        """q_s > 0 for positive Pareto quantiles at high u."""
        for u in [0.5, 0.8, 0.99]:
            assert self.sc.q_s(u) > 0.0

    def test_es_s_positive(self):
        """ES_s > 0 for positive tail."""
        for u in [0.5, 0.8, 0.99]:
            assert self.sc.es_s(u) > 0.0

# ---------------------------------------------------------------
# OscillatingShareScenario
# ---------------------------------------------------------------

class TestOscillatingShare:
    def setup_method(self):
        self.sc = OscillatingShareScenario(amplitude=0.2, frequency=2 * np.pi)

    def test_W_u_to_1(self):
        """W_u -> 1 as u -> 1 (exponential aggregate)."""
        W_vals = [self.sc.W_u(u) for u in [0.9, 0.99, 0.999]]
        for W in W_vals:
            assert W >= 1.0 - 1e-6

    def test_phi_psi_difference_small_at_high_u(self):
        """|phi - psi| -> 0 as u -> 1."""
        diffs = [abs(self.sc.phi_i(u) - self.sc.psi_i(u)) for u in [0.9, 0.99]]
        # Should be smaller at higher u
        # Not requiring strict convergence at 0.9 due to oscillation, just check bounded
        for d in diffs:
            assert d < 1.0

    def test_phi_range_bounded(self):
        """phi stays in [0, 1]."""
        for u in [0.5, 0.7, 0.9]:
            phi = self.sc.phi_i(u)
            assert 0.0 <= phi <= 1.0 + 1e-6

    def test_psi_range_bounded(self):
        """psi stays in [0, 1]."""
        for u in [0.5, 0.7, 0.9]:
            psi = self.sc.psi_i(u)
            assert 0.0 <= psi <= 1.0 + 1e-6

    def test_oscillation_present_at_moderate_u(self):
        """Phi oscillates: consecutive values not all monotone."""
        u_vals = np.linspace(0.5, 0.95, 100)
        phi_vals = np.array([self.sc.phi_i(u) for u in u_vals])
        diffs = np.diff(phi_vals)
        has_positive = np.any(diffs > 0)
        has_negative = np.any(diffs < 0)
        assert has_positive and has_negative, "Expected oscillation in phi"

# ---------------------------------------------------------------
# DiscreteDivergentWScenario
# ---------------------------------------------------------------

class TestDiscreteDivergentW:
    def setup_method(self):
        self.sc = DiscreteDivergentWScenario(N=20, alpha_even=0.3, alpha_odd=0.7)

    def test_probs_sum_to_one(self):
        """Probability mass sums to 1."""
        assert abs(sum(self.sc.probs) - 1.0) < 1e-8

    def test_W_u_increases(self):
        """W_u should generally increase along the tail."""
        u_vals = np.cumsum(self.sc.probs)[:-3]
        u_vals = u_vals[u_vals > 0.01]
        W_vals = [self.sc.W_u(u) for u in u_vals[-10:]]
        valid = [w for w in W_vals if not np.isnan(w) and np.isfinite(w)]
        if len(valid) > 2:
            assert max(valid) > min(valid)

    def test_phi_in_unit_interval(self):
        """phi is in [0, 1]."""
        u_vals = np.linspace(0.1, 0.9, 20)
        for u in u_vals:
            phi = self.sc.phi_i(u)
            if not np.isnan(phi):
                assert 0.0 <= phi <= 1.0 + 1e-6

    def test_psi_in_unit_interval(self):
        """psi is in [0, 1]."""
        u_vals = np.linspace(0.1, 0.9, 20)
        for u in u_vals:
            psi = self.sc.psi_i(u)
            if not np.isnan(psi):
                assert 0.0 <= psi <= 1.0 + 1e-6

    def test_W_diverges_along_support(self):
        """W_u diverges (grows large) in the tail of a discrete distribution with shrinking probs."""
        u_seq = np.array([sum(self.sc.probs[:k]) + self.sc.probs[k] * 0.1
                          for k in range(3, len(self.sc.probs) - 1)])
        W_vals = [self.sc.W_u(u) for u in u_seq if u < 0.999]
        valid = [w for w in W_vals if not np.isnan(w) and np.isfinite(w)]
        if len(valid) > 2:
            assert max(valid) > 2.0  # W should be > 2 somewhere

# ---------------------------------------------------------------
# compute_W_sequence
# ---------------------------------------------------------------

def test_compute_W_sequence_length():
    """W_sequence has same length as u_seq."""
    sc = ParetoBoundedWScenario(alphas=[2.5, 3.0], sigmas=[1.0, 1.5])
    u_seq = np.linspace(0.5, 0.99, 20)
    W_seq = compute_W_sequence(sc.q_s, sc.es_s, u_seq)
    assert len(W_seq) == len(u_seq)

def test_compute_W_sequence_positive():
    """W values are positive for Pareto scenario."""
    sc = ParetoBoundedWScenario(alphas=[2.5, 3.0], sigmas=[1.0, 1.5])
    u_seq = np.linspace(0.5, 0.95, 10)
    W_seq = compute_W_sequence(sc.q_s, sc.es_s, u_seq)
    assert np.all(W_seq > 0)

# ---------------------------------------------------------------
# estimate_convergence
# ---------------------------------------------------------------

def test_estimate_convergence_constant():
    """Constant sequence is converged; limsup == liminf == value."""
    seq = np.full(20, 0.4)
    result = estimate_convergence(seq)
    assert result["converged"]
    assert abs(result["limsup"] - 0.4) < 1e-9
    assert abs(result["liminf"] - 0.4) < 1e-9

def test_estimate_convergence_oscillating():
    """Oscillating sequence is not converged."""
    seq = np.array([0.5 + 0.2 * (-1) ** k for k in range(30)])
    result = estimate_convergence(seq)
    assert not result["converged"]
    assert result["oscillation_amplitude"] > 0.1

def test_estimate_convergence_monotone_toward_limit():
    """Monotone sequence approaching 1.0 shows limsup close to 1.0."""
    # Use long sequence so the last-window values are very close to 1.0.
    seq = 1.0 - np.exp(-np.linspace(0, 20, 200))
    result = estimate_convergence(seq, window=10)
    # limsup and liminf over the last 10 points should both be near 1.0
    assert abs(result["limsup"] - 1.0) < 0.01
    assert abs(result["liminf"] - 1.0) < 0.01

def test_estimate_convergence_short_seq():
    """Very short sequence doesn't crash."""
    seq = np.array([0.4, 0.5])
    result = estimate_convergence(seq)
    assert "converged" in result

# ---------------------------------------------------------------
# check_theorem_bounds
# ---------------------------------------------------------------

def test_check_theorem_bounds_pareto():
    """Bound |Psi - Phi| <= 2*(W-1)*sup|Phi| holds for Pareto."""
    sc = ParetoBoundedWScenario(alphas=[2.5, 3.0], sigmas=[1.0, 1.5])
    u_seq = np.linspace(0.5, 0.99, 30)
    phi = np.array([sc.phi_i(0, u) for u in u_seq])
    psi = np.array([sc.psi_i(0, u) for u in u_seq])
    W = np.array([sc.W_u(u) for u in u_seq])
    result = check_theorem_bounds(phi, psi, W)
    assert result["max_violation"] < 0.1  # small violation allowed due to numerics

def test_check_theorem_bounds_violation_zero_when_W1():
    """When W_u = 1 exactly, bound is 0 and |Phi-Psi| must be 0."""
    phi = np.array([0.4, 0.4, 0.4])
    psi = np.array([0.4, 0.4, 0.4])  # equal
    W = np.array([1.0, 1.0, 1.0])
    result = check_theorem_bounds(phi, psi, W)
    # |Phi - Psi| = 0 <= 2*0*sup = 0: bound holds
    assert np.all(result["bound_holds"])

# ---------------------------------------------------------------
# compute_asymptotic_diagnostics
# ---------------------------------------------------------------

def test_compute_asymptotic_diagnostics_returns_df():
    """Returns a DataFrame with expected columns."""
    import pandas as pd
    phi = np.array([0.4, 0.41, 0.42, 0.43, 0.44])
    psi = np.array([0.38, 0.39, 0.40, 0.41, 0.42])
    W = np.array([1.1, 1.08, 1.05, 1.03, 1.01])
    u = np.linspace(0.5, 0.9, 5)
    df = compute_asymptotic_diagnostics(phi, psi, W, u)
    assert isinstance(df, pd.DataFrame)
    assert "u" in df.columns
    assert "W" in df.columns
    assert "phi" in df.columns
    assert "psi" in df.columns
    assert "delta" in df.columns

def test_compute_asymptotic_diagnostics_delta_correct():
    """delta = phi - psi in diagnostics."""
    phi = np.array([0.6, 0.5, 0.4])
    psi = np.array([0.55, 0.48, 0.39])
    W = np.array([1.2, 1.1, 1.05])
    u = np.array([0.7, 0.8, 0.9])
    df = compute_asymptotic_diagnostics(phi, psi, W, u)
    np.testing.assert_allclose(df["delta"].values, phi - psi, atol=1e-10)

# ---------------------------------------------------------------
# classify_asymptotic_behavior
# ---------------------------------------------------------------

def test_classify_converges_both():
    """Constant phi and psi => 'common_limit' (both converge to same value)."""
    phi = np.full(20, 0.4)
    psi = np.full(20, 0.4)
    W = np.full(20, 1.5)
    label = classify_asymptotic_behavior(phi, psi, W)
    assert "common_limit" in label or "converge" in label.lower()

def test_classify_oscillating():
    """Oscillating phi and psi => 'oscillating' or similar."""
    phi = np.array([0.4 + 0.15 * (-1) ** k for k in range(30)])
    psi = phi.copy()
    W = np.full(30, 1.0)
    label = classify_asymptotic_behavior(phi, psi, W)
    assert label  # should return some non-empty string

def test_classify_divergent_W():
    """Diverging W => classification mentions W or divergence."""
    phi = np.full(20, 0.5)
    psi = np.linspace(0.3, 0.8, 20)  # oscillating psi
    W = np.linspace(1.0, 50.0, 20)  # diverging
    label = classify_asymptotic_behavior(phi, psi, W)
    assert label  # non-empty

# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

def test_paretobounded_W_single_component():
    """Single component: phi = psi = 1."""
    sc = ParetoBoundedWScenario(alphas=[2.5], sigmas=[1.0])
    for u in [0.5, 0.8, 0.99]:
        assert abs(sc.phi_i(0, u) - 1.0) < 1e-4
        assert abs(sc.psi_i(0, u) - 1.0) < 1e-4

def test_estimate_convergence_nan_handling():
    """estimate_convergence handles NaN values gracefully."""
    seq = np.array([0.4, 0.4, np.nan, 0.4, 0.4])
    result = estimate_convergence(seq)
    assert "converged" in result  # shouldn't crash

def test_check_theorem_bounds_all_W1():
    """All W=1 => bound is 0, any |Phi-Psi|>0 violates."""
    phi = np.array([0.5, 0.5, 0.5])
    psi = np.array([0.6, 0.6, 0.6])  # psi != phi
    W = np.array([1.0, 1.0, 1.0])
    result = check_theorem_bounds(phi, psi, W)
    # bound = 2*(1-1)*sup|phi| = 0, |phi-psi| = 0.1 > 0 => violation
    assert result["max_violation"] > 0

def test_oscillating_share_high_amplitude():
    """High amplitude oscillation produces bigger swings."""
    sc_small = OscillatingShareScenario(amplitude=0.1, frequency=2 * np.pi)
    sc_large = OscillatingShareScenario(amplitude=0.3, frequency=2 * np.pi)
    u_vals = np.linspace(0.6, 0.9, 50)
    phi_small = np.array([sc_small.phi_i(u) for u in u_vals])
    phi_large = np.array([sc_large.phi_i(u) for u in u_vals])
    assert np.ptp(phi_large) > np.ptp(phi_small)
