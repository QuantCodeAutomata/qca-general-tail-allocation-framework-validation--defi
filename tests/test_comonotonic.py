"""Tests for src/comonotonic.py — comonotonic allocation formulas."""

import numpy as np
import pytest
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.comonotonic import (
    compute_comonotonic_aggregate_quantile,
    compute_comonotonic_aggregate_quantile_grid,
    compute_phi_comonotonic,
    compute_psi_comonotonic,
    compute_h_ratio,
    compute_es_comonotonic,
    compute_hs_u,
    compute_location_scale_discrepancy,
    compute_pareto_quantile,
    compute_pareto_asymptotic_allocation,
    compute_allocations_on_grid,
    check_proportional_quantile_equality,
    check_ratio_monotonicity,
    build_pareto_scenario,
    LocationScaleScenario,
)

# Fixtures: simple proportional quantile example
@pytest.fixture
def proportional_funcs():
    """q_Xi = a_i * Normal(2,1) ppf; q_S = (a1+a2) * Normal(2,1) ppf."""
    a1, a2 = 0.4, 0.6
    q_x1 = lambda u: a1 * float(stats.norm.ppf(u, loc=2.0, scale=1.0))
    q_x2 = lambda u: a2 * float(stats.norm.ppf(u, loc=2.0, scale=1.0))
    return [q_x1, q_x2], a1, a2

# ---------------------------------------------------------------
# Aggregate quantile identity
# ---------------------------------------------------------------

def test_aggregate_quantile_identity(proportional_funcs):
    """q_S(u) = sum_i q_Xi(u)."""
    q_xi_funcs, a1, a2 = proportional_funcs
    a_s = a1 + a2
    for u in [0.3, 0.5, 0.7, 0.9]:
        q_s = compute_comonotonic_aggregate_quantile(q_xi_funcs, u)
        q_s_expected = a_s * float(stats.norm.ppf(u, loc=2.0, scale=1.0))
        assert abs(q_s - q_s_expected) < 1e-10

def test_aggregate_quantile_grid(proportional_funcs):
    """Vectorized aggregate quantile matches scalar version."""
    q_xi_funcs, _, _ = proportional_funcs
    u_grid = np.linspace(0.6, 0.98, 20)
    q_s_grid = compute_comonotonic_aggregate_quantile_grid(q_xi_funcs, u_grid)
    for i, u in enumerate(u_grid):
        q_s_scalar = compute_comonotonic_aggregate_quantile(q_xi_funcs, u)
        assert abs(q_s_grid[i] - q_s_scalar) < 1e-10

def test_aggregate_quantile_monotone(proportional_funcs):
    """Aggregate quantile is non-decreasing in u."""
    q_xi_funcs, _, _ = proportional_funcs
    u_grid = np.linspace(0.6, 0.98, 50)
    q_s_grid = compute_comonotonic_aggregate_quantile_grid(q_xi_funcs, u_grid)
    assert np.all(np.diff(q_s_grid) >= -1e-8)

# ---------------------------------------------------------------
# Phi and Psi
# ---------------------------------------------------------------

def test_phi_psi_proportional_equality(proportional_funcs):
    """Phi = Psi = a_i/(a1+a2) for proportional quantile."""
    q_xi_funcs, a1, a2 = proportional_funcs
    for u in [0.6, 0.75, 0.88]:
        phi1 = compute_phi_comonotonic(q_xi_funcs[0], q_xi_funcs, u, n_points=3000)
        psi1 = compute_psi_comonotonic(q_xi_funcs[0], q_xi_funcs, u, n_points=3000)
        expected = a1 / (a1 + a2)
        assert abs(phi1 - expected) < 1e-4, f"u={u}: phi={phi1:.5f}, expected={expected:.5f}"
        assert abs(psi1 - expected) < 1e-4, f"u={u}: psi={psi1:.5f}, expected={expected:.5f}"
        assert abs(phi1 - psi1) < 1e-4

def test_phi_psi_sum_to_one(proportional_funcs):
    """Phi1 + Phi2 = 1 and Psi1 + Psi2 = 1."""
    q_xi_funcs, _, _ = proportional_funcs
    for u in [0.6, 0.75]:
        phi1 = compute_phi_comonotonic(q_xi_funcs[0], q_xi_funcs, u)
        phi2 = compute_phi_comonotonic(q_xi_funcs[1], q_xi_funcs, u)
        psi1 = compute_psi_comonotonic(q_xi_funcs[0], q_xi_funcs, u)
        psi2 = compute_psi_comonotonic(q_xi_funcs[1], q_xi_funcs, u)
        assert abs(phi1 + phi2 - 1.0) < 1e-4
        assert abs(psi1 + psi2 - 1.0) < 1e-4

def test_phi_non_negative_for_positive_quantiles():
    """Phi >= 0 when quantiles are positive."""
    q_x1 = lambda u: compute_pareto_quantile(2.5, 1.0, u)
    q_x2 = lambda u: compute_pareto_quantile(3.0, 1.5, u)
    for u in [0.5, 0.7, 0.9]:
        phi1 = compute_phi_comonotonic(q_x1, [q_x1, q_x2], u)
        assert phi1 >= 0.0

# ---------------------------------------------------------------
# h(u) ratio
# ---------------------------------------------------------------

def test_h_ratio_constant_for_proportional(proportional_funcs):
    """h(u) = a_i / sum(a_j) = constant for proportional quantile."""
    q_xi_funcs, a1, a2 = proportional_funcs
    u_grid = np.linspace(0.6, 0.98, 30)
    h = compute_h_ratio(q_xi_funcs[0], q_xi_funcs, u_grid)
    valid = h[~np.isnan(h)]
    expected = a1 / (a1 + a2)
    assert np.allclose(valid, expected, atol=1e-8)

def test_h_ratio_monotone_pareto():
    """h(u) = q_X1(u)/q_S(u) is increasing when alpha1 < alpha2."""
    # X_1 ~ Pareto(2): heavier tail, X_2 ~ Pareto(4): lighter tail
    q_x1 = lambda u: compute_pareto_quantile(2.0, 1.0, u)
    q_x2 = lambda u: compute_pareto_quantile(4.0, 1.0, u)
    u_grid = np.linspace(0.5, 0.98, 40)
    h = compute_h_ratio(q_x1, [q_x1, q_x2], u_grid)
    valid = h[~np.isnan(h)]
    assert np.all(np.diff(valid) >= -1e-8), "h should be non-decreasing for heavier-tailed X1"

# ---------------------------------------------------------------
# ES and HS
# ---------------------------------------------------------------

def test_es_ge_quantile():
    """ES_u(S) >= VaR_u(S)."""
    q_x1 = lambda u: compute_pareto_quantile(2.5, 1.0, u)
    q_x2 = lambda u: compute_pareto_quantile(3.0, 1.5, u)
    q_xi_funcs = [q_x1, q_x2]
    for u in [0.5, 0.7, 0.9]:
        es = compute_es_comonotonic(q_xi_funcs, u, n_points=3000)
        qs = compute_comonotonic_aggregate_quantile(q_xi_funcs, u)
        assert es >= qs - 1e-6

def test_hs_le_es():
    """Harmonic mean HS_u(S) <= arithmetic mean ES_u(S) (AM >= HM)."""
    q_x1 = lambda u: compute_pareto_quantile(2.5, 1.0, u)
    q_x2 = lambda u: compute_pareto_quantile(3.0, 1.5, u)
    q_xi_funcs = [q_x1, q_x2]
    for u in [0.5, 0.7]:
        es = compute_es_comonotonic(q_xi_funcs, u, n_points=3000)
        hs = compute_hs_u(q_xi_funcs, u, n_points=3000)
        assert hs <= es + 1e-6

# ---------------------------------------------------------------
# Location-scale discrepancy
# ---------------------------------------------------------------

def test_location_scale_discrepancy_zero_at_equality():
    """Discrepancy = 0 when mu_i/sigma_i = mu_S/sigma_S."""
    # mu_1/sigma_1 = mu_S/sigma_S => sigma_i*mu_S - sigma_S*mu_i = 0
    mus = np.array([1.0, 2.0])
    sigmas = np.array([1.0, 2.0])  # ratio mu/sigma = 1 for both
    mu_s = 3.0
    sigma_s = 3.0
    disc = compute_location_scale_discrepancy(
        mu_i=1.0, sigma_i=1.0, mu_s=mu_s, sigma_s=sigma_s, es_u=2.0, hs_u=1.5
    )
    assert abs(disc) < 1e-10

def test_location_scale_discrepancy_formula():
    """Discrepancy matches formula: (sigma_i*mu_S - sigma_S*mu_i)/sigma_S * (1/HS - 1/ES)."""
    mu_i, sigma_i = 1.0, 1.0
    mu_s, sigma_s = 3.0, 2.0
    es_u, hs_u = 4.0, 3.0
    expected = (sigma_i * mu_s - sigma_s * mu_i) / sigma_s * (1.0 / hs_u - 1.0 / es_u)
    result = compute_location_scale_discrepancy(mu_i, sigma_i, mu_s, sigma_s, es_u, hs_u)
    assert abs(result - expected) < 1e-12

# ---------------------------------------------------------------
# Pareto quantile and asymptotic allocation
# ---------------------------------------------------------------

def test_pareto_quantile_formula():
    """q_X(u) = sigma * (1-u)^{-1/alpha}."""
    for alpha, sigma in [(2.5, 1.0), (3.0, 2.0)]:
        for u in [0.5, 0.8, 0.95]:
            expected = sigma * (1 - u) ** (-1.0 / alpha)
            result = compute_pareto_quantile(alpha, sigma, u)
            assert abs(result - expected) < 1e-12

def test_pareto_asymptotic_equal_alpha():
    """Equal alpha: asymptotic limit = sigma_i / sum(sigma_j)."""
    alphas = np.array([2.5, 2.5])
    sigmas = np.array([1.0, 2.0])
    expected = 1.0 / 3.0
    result = compute_pareto_asymptotic_allocation(alphas, sigmas, 0)
    assert abs(result - expected) < 1e-10

def test_pareto_asymptotic_unequal_alpha():
    """Lighter tail (larger alpha) has limit 0; heavier tail has limit 1."""
    alphas = np.array([2.0, 4.0])
    sigmas = np.array([1.0, 1.0])
    assert abs(compute_pareto_asymptotic_allocation(alphas, sigmas, 0) - 1.0) < 1e-10
    assert abs(compute_pareto_asymptotic_allocation(alphas, sigmas, 1) - 0.0) < 1e-10

def test_pareto_asymptotic_multiple_min():
    """Multiple min-alpha: split by sigma within min-alpha subset."""
    alphas = np.array([2.0, 2.0, 4.0])
    sigmas = np.array([1.0, 2.0, 1.0])
    lim0 = compute_pareto_asymptotic_allocation(alphas, sigmas, 0)
    lim1 = compute_pareto_asymptotic_allocation(alphas, sigmas, 1)
    lim2 = compute_pareto_asymptotic_allocation(alphas, sigmas, 2)
    assert abs(lim0 - 1.0 / 3.0) < 1e-10
    assert abs(lim1 - 2.0 / 3.0) < 1e-10
    assert abs(lim2 - 0.0) < 1e-10
    assert abs(lim0 + lim1 + lim2 - 1.0) < 1e-10

def test_pareto_asymptotic_sum_to_one():
    """Asymptotic limits always sum to 1."""
    alphas = np.array([2.0, 2.5, 3.0])
    sigmas = np.array([1.0, 1.0, 1.0])
    total = sum(compute_pareto_asymptotic_allocation(alphas, sigmas, i) for i in range(3))
    assert abs(total - 1.0) < 1e-10

# ---------------------------------------------------------------
# compute_allocations_on_grid
# ---------------------------------------------------------------

def test_allocations_on_grid_phi_sum(proportional_funcs):
    """sum_i Phi_{i,u} = 1 for all u on grid."""
    q_xi_funcs, _, _ = proportional_funcs
    u_grid = np.linspace(0.6, 0.95, 20)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=2000)
    phi = result["phi"]
    np.testing.assert_allclose(phi.sum(axis=0), 1.0, atol=1e-3)

def test_allocations_on_grid_psi_sum(proportional_funcs):
    """sum_i Psi_{i,u} = 1 for all u on grid."""
    q_xi_funcs, _, _ = proportional_funcs
    u_grid = np.linspace(0.6, 0.95, 20)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=2000)
    psi = result["psi"]
    np.testing.assert_allclose(psi.sum(axis=0), 1.0, atol=1e-3)

def test_allocations_on_grid_delta_zero_proportional(proportional_funcs):
    """Delta = Phi - Psi = 0 for proportional quantile."""
    q_xi_funcs, _, _ = proportional_funcs
    u_grid = np.linspace(0.6, 0.95, 20)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=3000)
    delta = result["delta"]
    assert np.nanmax(np.abs(delta)) < 1e-4

# ---------------------------------------------------------------
# Monotonicity and equality checks
# ---------------------------------------------------------------

def test_check_proportional_equality_true(proportional_funcs):
    """Proportional quantile passes equality check."""
    q_xi_funcs, _, _ = proportional_funcs
    u_grid = np.linspace(0.6, 0.95, 20)
    is_equal = check_proportional_quantile_equality(q_xi_funcs, u_grid)
    assert is_equal

def test_check_proportional_equality_false_pareto():
    """Non-proportional Pareto fails equality check."""
    q_x1 = lambda u: compute_pareto_quantile(2.0, 1.0, u)
    q_x2 = lambda u: compute_pareto_quantile(4.0, 1.0, u)
    u_grid = np.linspace(0.5, 0.95, 20)
    is_equal = check_proportional_quantile_equality([q_x1, q_x2], u_grid)
    assert not is_equal

def test_check_ratio_monotonicity_increasing():
    """Increasing h array returns monotone_increasing=True."""
    u_grid = np.linspace(0.5, 0.9, 20)
    h = np.linspace(0.2, 0.8, 20)  # strictly increasing
    result = check_ratio_monotonicity(h, u_grid)
    assert result["monotone_increasing"]
    assert not result["monotone_decreasing"]
    assert not result["is_constant"]

def test_check_ratio_monotonicity_constant():
    """Constant h returns is_constant=True."""
    u_grid = np.linspace(0.5, 0.9, 20)
    h = np.full(20, 0.4)
    result = check_ratio_monotonicity(h, u_grid)
    assert result["is_constant"]

# ---------------------------------------------------------------
# build_pareto_scenario
# ---------------------------------------------------------------

def test_build_pareto_scenario_returns_callables():
    """build_pareto_scenario returns correct number of callables."""
    alphas = [2.0, 3.0, 4.0]
    sigmas = [1.0, 1.5, 0.5]
    q_xi_funcs, ppf_list = build_pareto_scenario(alphas, sigmas)
    assert len(q_xi_funcs) == 3
    assert len(ppf_list) == 3

def test_build_pareto_scenario_values():
    """Pareto quantile functions return expected values."""
    alphas = [2.5]
    sigmas = [1.0]
    q_xi_funcs, _ = build_pareto_scenario(alphas, sigmas)
    for u in [0.5, 0.8]:
        expected = compute_pareto_quantile(2.5, 1.0, u)
        assert abs(q_xi_funcs[0](u) - expected) < 1e-10

# ---------------------------------------------------------------
# LocationScaleScenario
# ---------------------------------------------------------------

class TestLocationScaleScenario:
    def setup_method(self):
        self.mus = np.array([1.0, 2.0])
        self.sigmas = np.array([1.0, 2.0])
        self.z_func = lambda u: float(stats.norm.ppf(u))
        self.sc = LocationScaleScenario(mus=self.mus, sigmas=self.sigmas, z_func=self.z_func)

    def test_q_xi_formula(self):
        """q_Xi(u) = mu_i + sigma_i * z(u)."""
        for u in [0.6, 0.8, 0.9]:
            z = self.z_func(u)
            for i in range(2):
                expected = self.mus[i] + self.sigmas[i] * z
                result = self.sc.q_xi(i, u)
                assert abs(result - expected) < 1e-10

    def test_q_s_formula(self):
        """q_S(u) = mu_S + sigma_S * z(u)."""
        for u in [0.6, 0.8, 0.9]:
            z = self.z_func(u)
            expected = self.sc.mu_s + self.sc.sigma_s * z
            result = self.sc.q_s(u)
            assert abs(result - expected) < 1e-10

    def test_mu_s_sigma_s(self):
        """mu_S and sigma_S are sums of components."""
        assert abs(self.sc.mu_s - self.mus.sum()) < 1e-10
        assert abs(self.sc.sigma_s - self.sigmas.sum()) < 1e-10

    def test_q_s_equals_sum_q_xi(self):
        """q_S(u) = q_X1(u) + q_X2(u)."""
        for u in [0.6, 0.75, 0.9]:
            q_s = self.sc.q_s(u)
            q_sum = self.sc.q_xi(0, u) + self.sc.q_xi(1, u)
            assert abs(q_s - q_sum) < 1e-10

# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

def test_phi_returns_nan_when_qS_zero():
    """Phi returns nan/0 when q_S = 0 everywhere."""
    q_x1 = lambda u: 0.0
    q_x2 = lambda u: 0.0
    phi = compute_phi_comonotonic(q_x1, [q_x1, q_x2], 0.7)
    assert np.isnan(phi) or phi == 0.0

def test_pareto_quantile_extreme_u():
    """Pareto quantile grows to infinity as u -> 1."""
    alpha, sigma = 2.5, 1.0
    q_99 = compute_pareto_quantile(alpha, sigma, 0.99)
    q_999 = compute_pareto_quantile(alpha, sigma, 0.999)
    assert q_999 > q_99
    assert q_999 > 1.0
