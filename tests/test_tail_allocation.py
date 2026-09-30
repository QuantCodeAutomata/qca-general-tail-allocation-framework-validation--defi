"""Tests for src/tail_allocation.py — core framework objects."""

import numpy as np
import pytest
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.tail_allocation import (
    compute_left_quantile,
    compute_quantile_grid,
    compute_es,
    compute_W_u,
    compute_theta_u,
    compute_pi_u,
    compute_r_i,
    compute_phi_continuous,
    compute_psi_continuous,
    compute_phi_quantile,
    compute_psi_quantile,
    compute_h,
    compute_covariance_identity,
    compute_effective_tail_domain,
    validate_allocations,
    TailAllocationScenario,
)

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def standard_normal_cdf():
    return lambda x: float(stats.norm.cdf(x))

@pytest.fixture
def u_grid():
    return np.linspace(0.1, 0.99, 50)

# ---------------------------------------------------------------
# compute_left_quantile
# ---------------------------------------------------------------

def test_left_quantile_normal_exact():
    """Left-continuous quantile matches scipy.norm.ppf for continuous dist."""
    cdf = lambda x: float(stats.norm.cdf(x))
    for u in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        q = compute_left_quantile(cdf, u, -10.0, 10.0)
        expected = stats.norm.ppf(u)
        assert abs(q - expected) < 1e-6, f"u={u}: q={q:.6f}, expected={expected:.6f}"

def test_left_quantile_edge_u0(standard_normal_cdf):
    """u=0 returns x_min."""
    q = compute_left_quantile(standard_normal_cdf, 0.0, -10.0, 10.0)
    assert q == -10.0

def test_left_quantile_edge_u1(standard_normal_cdf):
    """u=1 returns x_max."""
    q = compute_left_quantile(standard_normal_cdf, 1.0, -10.0, 10.0)
    assert q == 10.0

def test_left_quantile_gamma():
    """Left-continuous quantile for Gamma distribution."""
    cdf = lambda x: float(stats.gamma.cdf(x, a=2.0, scale=1.0))
    for u in [0.3, 0.6, 0.9]:
        q = compute_left_quantile(cdf, u, 0.0, 30.0)
        expected = stats.gamma.ppf(u, a=2.0, scale=1.0)
        assert abs(q - expected) < 1e-5, f"u={u}: q={q:.6f}, expected={expected:.6f}"

def test_left_quantile_discrete_left_continuous():
    """Discrete CDF: left-continuous quantile is the exact atom."""
    # Bernoulli(0.3): CDF jumps at 0 and 1
    probs = {0: 0.7, 1: 0.3}
    cdf = lambda x: (0.0 if x < 0 else (0.7 if x < 1 else 1.0))
    # q(0.5) = 0 (F(0) = 0.7 >= 0.5, so inf{x: F(x) >= 0.5} = 0)
    q = compute_left_quantile(cdf, 0.5, -1.0, 2.0, tol=1e-3)
    assert abs(q - 0.0) < 0.1

# ---------------------------------------------------------------
# compute_quantile_grid
# ---------------------------------------------------------------

def test_quantile_grid_shape(standard_normal_cdf):
    """Grid has same length as u_grid."""
    u_grid = np.linspace(0.1, 0.9, 30)
    q_grid = compute_quantile_grid(standard_normal_cdf, u_grid, -10.0, 10.0)
    assert len(q_grid) == len(u_grid)

def test_quantile_grid_monotone(standard_normal_cdf):
    """Quantiles are non-decreasing in u."""
    u_grid = np.linspace(0.1, 0.9, 30)
    q_grid = compute_quantile_grid(standard_normal_cdf, u_grid, -10.0, 10.0)
    assert np.all(np.diff(q_grid) >= -1e-10)

# ---------------------------------------------------------------
# compute_es
# ---------------------------------------------------------------

def test_es_normal_closed_form():
    """ES for Normal(0,1) has closed form: ES_u = phi(z_u)/(1-u) where z_u = ppf(u)."""
    u_grid = np.linspace(0.5, 0.99, 100)
    q_grid = stats.norm.ppf(u_grid)
    es_grid = compute_es(q_grid, u_grid)
    # Closed form ES_u(Z) = phi(z_u)/(1-u)
    expected = stats.norm.pdf(q_grid) / (1 - u_grid)
    # Numerical integration on discrete grid has some error, use relative tolerance
    rel_err = np.abs(es_grid - expected) / np.abs(expected)
    assert np.mean(rel_err) < 0.05, f"Mean relative ES error: {np.mean(rel_err):.4f}"

def test_es_monotone():
    """ES should be non-decreasing in u (higher threshold -> higher tail mean)."""
    u_grid = np.linspace(0.1, 0.98, 100)
    q_grid = stats.norm.ppf(u_grid)
    es_grid = compute_es(q_grid, u_grid)
    assert np.all(np.diff(es_grid) >= -0.01)

def test_es_ge_quantile():
    """ES_u(S) >= VaR_u(S) by definition."""
    u_grid = np.linspace(0.3, 0.97, 50)
    q_grid = stats.norm.ppf(u_grid, loc=1.0, scale=1.5)
    es_grid = compute_es(q_grid, u_grid)
    # ES >= q(u) within numerical tolerance
    assert np.all(es_grid >= q_grid - 1e-6)

# ---------------------------------------------------------------
# compute_W_u
# ---------------------------------------------------------------

def test_W_u_ge_1():
    """W_u = ES_u(S) / q(u) >= 1 when q(u) > 0."""
    u_grid = np.linspace(0.5, 0.98, 50)
    q_grid = stats.norm.ppf(u_grid, loc=1.0)
    es_grid = compute_es(q_grid, u_grid)
    W = compute_W_u(es_grid, q_grid)
    valid = ~np.isnan(W) & (q_grid > 0)
    assert np.all(W[valid] >= 1.0 - 1e-6)

def test_W_u_nan_when_q_zero():
    """W_u = nan when q(u) = 0."""
    es = np.array([1.0, 2.0])
    q = np.array([0.0, 1.0])
    W = compute_W_u(es, q)
    assert np.isnan(W[0])
    assert abs(W[1] - 2.0) < 1e-10

# ---------------------------------------------------------------
# compute_theta_u
# ---------------------------------------------------------------

def test_theta_u_continuous_is_zero():
    """For a continuous distribution, theta_u = 0."""
    cdf = lambda x: float(stats.norm.cdf(x))
    theta = compute_theta_u(cdf, None, 0.0, 0.5)
    assert abs(theta) < 1e-10

def test_theta_u_bernoulli():
    """For Bernoulli(0.3), check theta_u at atom."""
    # F(0) = 0.7, F(1) = 1.0, P(X=0)=0.7, P(X=1)=0.3
    cdf = lambda x: 0.0 if x < 0 else (0.7 if x < 1 else 1.0)
    pmf = lambda x: 0.7 if abs(x) < 1e-10 else (0.3 if abs(x - 1) < 1e-10 else 0.0)
    # For u = 0.5: q(0.5) = 0, F(0) = 0.7, theta = (0.7 - 0.5) / 0.7
    theta = compute_theta_u(cdf, pmf, 0.0, 0.5)
    expected = (0.7 - 0.5) / 0.7
    assert abs(theta - expected) < 1e-8

def test_theta_u_in_unit_interval():
    """theta_u must be in [0, 1]."""
    cdf = lambda x: 0.0 if x < 0 else (0.7 if x < 1 else 1.0)
    pmf = lambda x: 0.7 if abs(x) < 1e-10 else (0.3 if abs(x - 1) < 1e-10 else 0.0)
    for u in [0.1, 0.3, 0.5, 0.65]:
        q_u = 0.0  # always hits atom at 0 for u < 0.7
        theta = compute_theta_u(cdf, pmf, q_u, u)
        assert 0.0 <= theta <= 1.0 + 1e-10

# ---------------------------------------------------------------
# compute_pi_u
# ---------------------------------------------------------------

def test_pi_u_continuous():
    """For continuous dist, pi_u(s) = 1_{s > q_u}."""
    s = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    q_u = 1.0
    pi = compute_pi_u(s, q_u, theta_u=0.0)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(pi, expected)

def test_pi_u_with_atom():
    """With theta_u = 0.5 and atom at q_u, pi_u(q_u) = 0.5."""
    s = np.array([0.9, 1.0, 1.1])
    q_u = 1.0
    pi = compute_pi_u(s, q_u, theta_u=0.5)
    assert abs(pi[0] - 0.0) < 1e-10
    assert abs(pi[1] - 0.5) < 1e-10
    assert abs(pi[2] - 1.0) < 1e-10

# ---------------------------------------------------------------
# compute_r_i
# ---------------------------------------------------------------

def test_r_i_sums_to_one():
    """sum_i R_i = 1 almost surely when sum(c_i) = 1 and S != 0."""
    rng = np.random.default_rng(0)
    X1 = rng.normal(1.0, 1.0, 1000)
    X2 = rng.normal(2.0, 1.0, 1000)
    S = X1 + X2
    R1 = compute_r_i(X1, S, c_i=0.3)
    R2 = compute_r_i(X2, S, c_i=0.7)
    mask = np.abs(S) > 1e-10
    np.testing.assert_allclose(R1[mask] + R2[mask], 1.0, atol=1e-10)

def test_r_i_uses_c_i_at_zero():
    """R_i = c_i when S = 0."""
    X1 = np.array([0.0, 1.0, 2.0])
    S = np.array([0.0, 3.0, 0.0])
    R1 = compute_r_i(X1, S, c_i=0.4)
    assert abs(R1[0] - 0.4) < 1e-10
    assert abs(R1[2] - 0.4) < 1e-10
    assert abs(R1[1] - 1.0 / 3.0) < 1e-10

# ---------------------------------------------------------------
# compute_phi_continuous, compute_psi_continuous
# ---------------------------------------------------------------

def test_phi_psi_sum_to_one():
    """sum_i Phi_{i,u} = 1 and sum_i Psi_{i,u} = 1."""
    rng = np.random.default_rng(42)
    X1 = rng.normal(1.0, 1.0, 100_000)
    X2 = rng.normal(2.0, 1.0, 100_000)
    S = X1 + X2
    R1 = compute_r_i(X1, S, c_i=0.33)
    R2 = compute_r_i(X2, S, c_i=0.67)
    q_u = float(np.percentile(S, 70))
    phi1 = compute_phi_continuous(X1, S, q_u, 0.0)
    phi2 = compute_phi_continuous(X2, S, q_u, 0.0)
    psi1 = compute_psi_continuous(R1, S, q_u, 0.0)
    psi2 = compute_psi_continuous(R2, S, q_u, 0.0)
    assert abs(phi1 + phi2 - 1.0) < 0.01
    assert abs(psi1 + psi2 - 1.0) < 0.01

def test_phi_psi_equal_for_constant_R():
    """When R_i is constant, Phi = Psi = constant."""
    rng = np.random.default_rng(5)
    Z = rng.normal(2.0, 1.0, 100_000)
    X1 = 0.4 * Z
    X2 = 0.6 * Z
    S = X1 + X2
    R1 = compute_r_i(X1, S, c_i=0.4)
    q_u = float(np.percentile(S, 60))
    phi1 = compute_phi_continuous(X1, S, q_u, 0.0)
    psi1 = compute_psi_continuous(R1, S, q_u, 0.0)
    assert abs(phi1 - 0.4) < 0.02
    assert abs(psi1 - 0.4) < 0.02
    assert abs(phi1 - psi1) < 0.02

def test_phi_psi_non_negative():
    """Phi and Psi should be >= 0 for positive risks."""
    rng = np.random.default_rng(7)
    X1 = rng.exponential(1.0, 10_000)
    X2 = rng.exponential(2.0, 10_000)
    S = X1 + X2
    R1 = compute_r_i(X1, S)
    q_u = float(np.percentile(S, 80))
    phi1 = compute_phi_continuous(X1, S, q_u, 0.0)
    psi1 = compute_psi_continuous(R1, S, q_u, 0.0)
    assert phi1 >= 0.0
    assert psi1 >= 0.0

# ---------------------------------------------------------------
# compute_phi_quantile, compute_psi_quantile
# ---------------------------------------------------------------

def test_phi_psi_quantile_sum_to_one():
    """Phi and Psi from quantile integrals sum to 1 (comonotonic case)."""
    # X_1 = a*Z, X_2 = b*Z, Z ~ N(2,1)
    a, b = 0.4, 0.6
    q_x1 = lambda u: a * stats.norm.ppf(u, loc=2.0, scale=1.0)
    q_x2 = lambda u: b * stats.norm.ppf(u, loc=2.0, scale=1.0)
    q_s = lambda u: (a + b) * stats.norm.ppf(u, loc=2.0, scale=1.0)
    u = 0.7
    phi1 = compute_phi_quantile(q_x1, q_s, u)
    phi2 = compute_phi_quantile(q_x2, q_s, u)
    assert abs(phi1 + phi2 - 1.0) < 1e-4

def test_phi_quantile_equals_ratio_for_proportional():
    """For q_Xi = a_i * f(u), Phi = a_i / sum(a_j)."""
    a = [0.3, 0.7]
    q_x1 = lambda u: a[0] * stats.norm.ppf(u, loc=2.0, scale=1.0)
    q_x2 = lambda u: a[1] * stats.norm.ppf(u, loc=2.0, scale=1.0)
    q_s = lambda u: sum(a) * stats.norm.ppf(u, loc=2.0, scale=1.0)
    for u in [0.6, 0.7, 0.8]:
        phi1 = compute_phi_quantile(q_x1, q_s, u)
        assert abs(phi1 - a[0] / sum(a)) < 1e-4, f"u={u}: phi={phi1:.4f}"

# ---------------------------------------------------------------
# compute_h
# ---------------------------------------------------------------

def test_h_constant_for_proportional():
    """For q_Xi = a_i * q_S, h(u) should be constant = a_i."""
    a = 0.4
    q_xi = lambda u: a * stats.norm.ppf(u, loc=2.0, scale=1.0)
    q_s_func = lambda u: stats.norm.ppf(u, loc=2.0, scale=1.0)
    u_grid = np.linspace(0.6, 0.98, 50)
    h = compute_h(q_xi, q_s_func, u_grid)
    valid = h[~np.isnan(h)]
    assert np.ptp(valid) < 1e-8, f"h not constant: range={np.ptp(valid):.2e}"

def test_h_nan_at_zero_qs():
    """h(u) = nan where q_S(u) = 0."""
    q_xi = lambda u: float(stats.norm.ppf(u))
    q_s_func = lambda u: 0.0
    u_grid = np.array([0.5])
    h = compute_h(q_xi, q_s_func, u_grid)
    assert np.isnan(h[0])

# ---------------------------------------------------------------
# compute_effective_tail_domain
# ---------------------------------------------------------------

def test_tail_domain_open_for_continuous():
    """Continuous distribution: lower endpoint is open."""
    cdf = lambda x: float(stats.norm.cdf(x))
    q_u_star = stats.norm.ppf(0.8)
    domain = compute_effective_tail_domain(cdf, q_u_star, 0.8, 10.0)
    assert domain["lower_open"] is True

def test_tail_domain_lower_value():
    """Lower value of tail domain equals q(u*)."""
    cdf = lambda x: float(stats.norm.cdf(x))
    q_u_star = stats.norm.ppf(0.7)
    domain = compute_effective_tail_domain(cdf, q_u_star, 0.7, 10.0)
    assert abs(domain["lower"] - q_u_star) < 1e-10

# ---------------------------------------------------------------
# validate_allocations
# ---------------------------------------------------------------

def test_validate_allocations_ok():
    """Valid allocations pass."""
    result = validate_allocations([0.3, 0.7], [0.4, 0.6])
    assert result["phi_ok"]
    assert result["psi_ok"]

def test_validate_allocations_fail():
    """Invalid allocations fail."""
    result = validate_allocations([0.3, 0.5], [0.4, 0.7])
    assert not result["phi_ok"]
    assert not result["psi_ok"]

# ---------------------------------------------------------------
# TailAllocationScenario
# ---------------------------------------------------------------

def test_tail_allocation_scenario_to_dataframe():
    """TailAllocationScenario.to_dataframe returns DataFrame with correct shape."""
    u_grid = np.linspace(0.5, 0.9, 10)
    q_grid = stats.norm.ppf(u_grid)
    es_grid = q_grid + 0.5
    W_grid = es_grid / q_grid
    sc = TailAllocationScenario(
        name="test",
        u_grid=u_grid,
        q_grid=q_grid,
        es_grid=es_grid,
        W_grid=W_grid,
        phi_results={"comp1": np.full(10, 0.4)},
        psi_results={"comp1": np.full(10, 0.4)},
        delta_results={"comp1": np.zeros(10)},
        h_results={"comp1": np.full(10, 0.4)},
    )
    df = sc.to_dataframe()
    assert len(df) == 10
    assert "u" in df.columns
    assert "phi_comp1" in df.columns

# ---------------------------------------------------------------
# covariance identity (spot check)
# ---------------------------------------------------------------

def test_covariance_identity_proportional():
    """For proportional quantile (h constant), covariance term = 0 = Phi - Psi."""
    a = 0.4
    q_s_func = lambda u: (a + 0.6) * stats.norm.ppf(u, loc=2.0, scale=1.0)
    h_func = lambda u: a  # constant

    result = compute_covariance_identity(q_s_func, h_func, 0.7)
    assert abs(result["delta_from_cov"]) < 1e-5, \
        f"Cov term should be ~0 for constant h, got {result['delta_from_cov']:.2e}"

# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

def test_phi_returns_nan_when_no_tail_mass():
    """Phi returns nan when no samples fall in the tail."""
    X1 = np.array([1.0, 2.0, 3.0])
    S = np.array([3.0, 5.0, 7.0])
    phi = compute_phi_continuous(X1, S, q_u=100.0, theta_u=0.0)  # threshold above all
    assert np.isnan(phi)

def test_es_single_element():
    """ES on a grid with one point doesn't crash."""
    u_grid = np.array([0.5])
    q_grid = np.array([0.0])
    es_grid = compute_es(q_grid, u_grid)
    assert len(es_grid) == 1
