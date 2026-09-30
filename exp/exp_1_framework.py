"""
Experiment 1: General Tail-Allocation Framework Validation

Validates:
- Realized risk shares R_i = X_i/S
- Effective tail domains mathfrak{S}_{u*}
- Randomized tail inclusion pi_u(s) with theta_u
- Proportional ES/GTE allocations Phi_{i,u}, Psi_{i,u}
- Covariance identity Phi - Psi = Cov_u(q(V), h(V)) / E_u[q(V)]
- Equality theorem (constant g_i => Phi = Psi)
- Dominance theorem (monotone g_i => sign of Phi - Psi)
- Localization necessity
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate
import os
import sys
import warnings

warnings.filterwarnings("ignore")

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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Scenario A: Continuous Normal model
# X_1 ~ N(mu1, sig1^2), X_2 ~ N(mu2, sig2^2), independent
# S = X_1 + X_2 ~ N(mu1+mu2, sig1^2+sig2^2)
# -------------------------------------------------------------------

def run_normal_scenario(
    mu1: float = 0.5, sig1: float = 1.0,
    mu2: float = 1.5, sig2: float = 1.0,
    u_star: float = 0.5,
    n_grid: int = 80,
    n_mc: int = 50_000,
    rng_seed: int = 42,
) -> dict:
    """Run continuous Normal scenario for Exp 1."""

    rng = np.random.default_rng(rng_seed)
    mu_s = mu1 + mu2
    sig_s = np.sqrt(sig1**2 + sig2**2)
    beta1 = sig1**2 / sig_s**2
    beta2 = sig2**2 / sig_s**2

    def cdf_s(x: float) -> float:
        return float(stats.norm.cdf(x, loc=mu_s, scale=sig_s))

    # u-grid on [u_star, 1); use norm.ppf directly for speed (no brentq)
    u_grid = np.linspace(u_star, 0.999, n_grid)
    q_grid = stats.norm.ppf(u_grid, loc=mu_s, scale=sig_s)
    es_grid = compute_es(q_grid, u_grid)
    W_grid = compute_W_u(es_grid, q_grid)

    # Effective tail domain
    q_u_star = float(stats.norm.ppf(u_star, loc=mu_s, scale=sig_s))
    tail_domain = compute_effective_tail_domain(cdf_s, q_u_star, u_star, mu_s + 8*sig_s)

    # theta_u for continuous: should be 0
    theta_vals = [compute_theta_u(cdf_s, None, q_grid[i], u_grid[i]) for i in range(n_grid)]
    assert all(abs(t) < 1e-10 for t in theta_vals), "theta_u should be 0 for continuous dist"

    # Vectorised exact conditional shares for Normal
    # g_i(s) = beta_i + (mu_i - beta_i * mu_S) / s  (with safe division)
    def g1_exact_vec(s: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(s) < 1e-12, np.ones_like(s), s)
        return np.where(np.abs(s) < 1e-12, beta1, beta1 + (mu1 - beta1 * mu_s) / safe)

    # Monte Carlo samples
    X1 = rng.normal(mu1, sig1, n_mc)
    X2 = rng.normal(mu2, sig2, n_mc)
    S = X1 + X2

    R1 = compute_r_i(X1, S, c_i=beta1)
    R2 = compute_r_i(X2, S, c_i=beta2)

    phi1_list, psi1_list, phi2_list, psi2_list = [], [], [], []

    for i in range(n_grid):
        q_u = q_grid[i]
        phi1 = compute_phi_continuous(X1, S, q_u, 0.0)
        psi1 = compute_psi_continuous(R1, S, q_u, 0.0)
        phi2 = compute_phi_continuous(X2, S, q_u, 0.0)
        psi2 = compute_psi_continuous(R2, S, q_u, 0.0)
        phi1_list.append(phi1)
        psi1_list.append(psi1)
        phi2_list.append(phi2)
        psi2_list.append(psi2)

    phi1_arr = np.array(phi1_list)
    psi1_arr = np.array(psi1_list)
    phi2_arr = np.array(phi2_list)
    psi2_arr = np.array(psi2_list)

    # Validate allocations sum to 1 (spot check every 10)
    for i in range(0, n_grid, 10):
        val = validate_allocations([phi1_arr[i], phi2_arr[i]], [psi1_arr[i], psi2_arr[i]])
        assert val["phi_ok"], f"Phi sum != 1 at u={u_grid[i]:.3f}: sum={val['phi_sum']:.6f}"
        assert val["psi_ok"], f"Psi sum != 1 at u={u_grid[i]:.3f}: sum={val['psi_sum']:.6f}"

    # Covariance identity: vectorised over sparse sub-grid
    cov_residuals = []
    for i in range(0, n_grid, 16):
        u = u_grid[i]
        # V ~ U(u, 0.9999) — draw once, vectorised ppf
        V_samp = rng.uniform(u, 0.9999, 20_000)
        q_V = stats.norm.ppf(V_samp, loc=mu_s, scale=sig_s)
        h_V = g1_exact_vec(q_V)
        cov_val = np.cov(q_V, h_V)[0, 1]
        eu_q = np.mean(q_V)
        cov_term = cov_val / eu_q if abs(eu_q) > 1e-12 else np.nan
        delta_direct = phi1_arr[i] - psi1_arr[i]
        if not np.isnan(cov_term):
            cov_residuals.append(abs(cov_term - delta_direct))

    # Equality check: Phi = Psi iff g_i constant (mu1 = beta1 * mu_s)
    mu1_eq = beta1 * mu_s
    X1_eq = rng.normal(mu1_eq, sig1, n_mc)
    X2_eq = rng.normal(mu_s - mu1_eq, sig2, n_mc)
    S_eq = X1_eq + X2_eq
    R1_eq = compute_r_i(X1_eq, S_eq, c_i=beta1)

    phi1_eq_list, psi1_eq_list = [], []
    for q_u in q_grid[::8]:
        phi1_eq_list.append(compute_phi_continuous(X1_eq, S_eq, q_u, 0.0))
        psi1_eq_list.append(compute_psi_continuous(R1_eq, S_eq, q_u, 0.0))

    equality_errors = [abs(p - q) for p, q in zip(phi1_eq_list, psi1_eq_list)]
    equality_holds = all(e < 0.02 for e in equality_errors)  # MC tolerance

    # Dominance: sign of Phi - Psi depends on mu1 vs beta1*mu_S
    # g_1(s) = beta_1 + (mu_1 - beta_1*mu_S)/s  =>  dg_1/ds = -(mu_1-beta_1*mu_S)/s^2
    # mu1 < beta1*mu_S => g_1 increasing => Phi_1 > Psi_1  (expected_sign = +1)
    # mu1 > beta1*mu_S => g_1 decreasing => Phi_1 < Psi_1  (expected_sign = -1)
    dom_sign = np.sign(phi1_arr[len(phi1_arr)//4:] - psi1_arr[len(psi1_arr)//4:])
    expected_sign = 1 if mu1 < beta1 * mu_s else -1
    dominance_holds = (np.sign(np.mean(dom_sign)) == expected_sign)

    # Localization: global Psi at u=0 (full distribution including negatives)
    # For Normal with signed support, when Q(0) = E[S] != 0 it may be defined
    # but if mu_s is near 0 or negative, Psi_{1,0} can be ill-posed
    # Demonstrate by trying u=0.01 where q(u) << 0 -> check Psi is valid
    global_u = 0.01
    q_global = float(stats.norm.ppf(global_u, loc=mu_s, scale=sig_s))
    mask_global = S > q_global
    psi1_global = (np.mean(R1[mask_global]) if mask_global.sum() > 100 else np.nan)
    phi1_global = (np.sum(X1[mask_global]) / np.sum(S[mask_global]) if mask_global.sum() > 100 else np.nan)
    # For localized u_star = 0.5 where q(u*) > 0, both are valid
    assert q_u_star > 0, f"q(u*)={q_u_star:.3f} should be > 0 for localization example"

    result = {
        "scenario": "Normal_Continuous",
        "mu1": mu1, "sig1": sig1, "mu2": mu2, "sig2": sig2,
        "mu_s": mu_s, "sig_s": sig_s, "beta1": beta1,
        "u_star": u_star, "q_u_star": q_u_star,
        "tail_domain": tail_domain,
        "theta_u_max": max(abs(t) for t in theta_vals),
        "equality_holds": equality_holds,
        "equality_max_err": max(equality_errors),
        "dominance_holds": dominance_holds,
        "expected_sign": expected_sign,
        "cov_identity_mean_err": float(np.mean(cov_residuals)) if cov_residuals else np.nan,
        "global_q": q_global,
        "phi1_global": phi1_global,
        "psi1_global": psi1_global,
        "phi1_u_star": float(np.nanmean(phi1_arr[:10])),
        "psi1_u_star": float(np.nanmean(psi1_arr[:10])),
        "u_grid": u_grid,
        "phi1": phi1_arr, "psi1": psi1_arr,
        "phi2": phi2_arr, "psi2": psi2_arr,
        "q_grid": q_grid, "es_grid": es_grid, "W_grid": W_grid,
    }
    return result


# -------------------------------------------------------------------
# Scenario B: Discrete Poisson model with atoms
# X_1 ~ Poisson(lambda1), X_2 ~ Poisson(lambda2), independent
# S ~ Poisson(lambda1 + lambda2)
# -------------------------------------------------------------------

def run_poisson_scenario(
    lambda1: float = 3.0,
    lambda2: float = 5.0,
    u_star: float = 0.5,
    s_max: int = 30,
    n_grid: int = 50,
) -> dict:
    """Run discrete Poisson scenario for Exp 1."""

    lam_s = lambda1 + lambda2
    s_vals = np.arange(0, s_max + 1)

    # Joint PMF via convolution: P(X1=k1, X2=k2) = Poisson(k1;lam1)*Poisson(k2;lam2)
    pmf1 = stats.poisson.pmf(s_vals, lambda1)
    pmf2 = stats.poisson.pmf(s_vals, lambda2)
    # P(S=s) for s in [0, s_max]
    pmf_s = np.convolve(pmf1, pmf2)[:s_max + 1]
    pmf_s = pmf_s / pmf_s.sum()  # normalize

    cdf_s_vals = np.cumsum(pmf_s)

    def cdf_s(s: float) -> float:
        idx = int(np.floor(s))
        if idx < 0:
            return 0.0
        if idx >= len(cdf_s_vals):
            return 1.0
        return float(cdf_s_vals[idx])

    def pmf_s_func(s: float) -> float:
        idx = int(round(s))
        if idx < 0 or idx >= len(pmf_s):
            return 0.0
        return float(pmf_s[idx])

    # Left-continuous quantile
    def q_s(u: float) -> float:
        idx = np.searchsorted(cdf_s_vals, u, side="left")
        return float(s_vals[min(idx, len(s_vals) - 1)])

    # u-grid
    u_grid_raw = np.linspace(u_star, 0.99, n_grid)
    q_grid = np.array([q_s(u) for u in u_grid_raw])

    # theta_u for atoms
    theta_list, nontrivial_theta = [], []
    for i, u in enumerate(u_grid_raw):
        q_u = q_grid[i]
        theta_u = compute_theta_u(cdf_s, pmf_s_func, q_u, u)
        theta_list.append(theta_u)
        if abs(pmf_s_func(q_u)) > 1e-10:
            nontrivial_theta.append(theta_u)

    # ES via tail expectation with pi_u
    es_list = []
    for i, u in enumerate(u_grid_raw):
        q_u = q_grid[i]
        theta_u = theta_list[i]
        # E[S * pi_u(S)] = sum_{s > q_u} s * p(s) + theta_u * q_u * p(q_u)
        mask_above = s_vals > q_u
        mask_at = np.abs(s_vals - q_u) < 0.5
        p_at = pmf_s[mask_at].sum()
        num = np.sum(s_vals[mask_above] * pmf_s[mask_above]) + theta_u * q_u * p_at
        # E[pi_u(S)] = 1 - u by construction
        denom = np.sum(pmf_s[mask_above]) + theta_u * p_at
        es_list.append(num / denom if denom > 1e-15 else np.nan)
    es_grid = np.array(es_list)
    W_grid = compute_W_u(es_grid, q_grid)

    # Conditional share: g1(s) = E[X1 | S=s]
    # E[X1 | S=s] = E[X1 * 1_{X1+X2=s}] / P(X1+X2=s)
    # = sum_{k=0}^{s} k * P(X1=k) * P(X2=s-k) / P(S=s)
    def g1_exact(s: int) -> float:
        if s < 0 or s > s_max:
            return lambda1 / lam_s
        s_int = int(s)
        pmf_s_val = pmf_s_func(s_int)
        if pmf_s_val < 1e-15:
            return lambda1 / lam_s
        numerator = sum(
            k * stats.poisson.pmf(k, lambda1) * stats.poisson.pmf(s_int - k, lambda2)
            for k in range(s_int + 1)
        )
        return numerator / pmf_s_val

    def g2_exact(s: int) -> float:
        return 1.0 - g1_exact(s)

    # R_i samples: compute over support
    r1_at_s = np.array([g1_exact(s) for s in s_vals])
    r2_at_s = 1.0 - r1_at_s

    # Compute Phi and Psi using discrete tail expectations
    phi1_list, psi1_list = [], []
    phi2_list, psi2_list = [], []

    for i, u in enumerate(u_grid_raw):
        q_u = q_grid[i]
        theta_u = theta_list[i]
        mask_above = s_vals > q_u
        mask_at = np.abs(s_vals - q_u) < 0.5
        p_at = pmf_s[mask_at].sum()

        # Phi1 = E[X1 * pi_u(S)] / E[S * pi_u(S)]
        # X1(s) = g1(s) * s
        x1_at_s = r1_at_s * s_vals
        x2_at_s = r2_at_s * s_vals
        x1_at = r1_at_s[mask_at].mean() * q_u if mask_at.any() else 0.0
        x2_at = r2_at_s[mask_at].mean() * q_u if mask_at.any() else 0.0
        s_at = q_u

        num1 = np.sum(x1_at_s[mask_above] * pmf_s[mask_above]) + theta_u * x1_at * p_at
        num2 = np.sum(x2_at_s[mask_above] * pmf_s[mask_above]) + theta_u * x2_at * p_at
        denom_s = np.sum(s_vals[mask_above] * pmf_s[mask_above]) + theta_u * s_at * p_at

        phi1_list.append(num1 / denom_s if abs(denom_s) > 1e-15 else np.nan)
        phi2_list.append(num2 / denom_s if abs(denom_s) > 1e-15 else np.nan)

        # Psi1 = E[R1 * pi_u(S)] / E[pi_u(S)]
        r1_at = r1_at_s[mask_at].mean() if mask_at.any() else lambda1 / lam_s
        r2_at = 1.0 - r1_at
        denom_pi = np.sum(pmf_s[mask_above]) + theta_u * p_at
        psi1_list.append(
            (np.sum(r1_at_s[mask_above] * pmf_s[mask_above]) + theta_u * r1_at * p_at)
            / denom_pi if abs(denom_pi) > 1e-15 else np.nan
        )
        psi2_list.append(
            (np.sum(r2_at_s[mask_above] * pmf_s[mask_above]) + theta_u * r2_at * p_at)
            / denom_pi if abs(denom_pi) > 1e-15 else np.nan
        )

    phi1_arr = np.array(phi1_list)
    psi1_arr = np.array(psi1_list)
    phi2_arr = np.array(phi2_list)
    psi2_arr = np.array(psi2_list)

    # Validate sums
    for i in range(0, len(u_grid_raw), 5):
        val = validate_allocations(
            [phi1_arr[i], phi2_arr[i]], [psi1_arr[i], psi2_arr[i]]
        )
        assert val["phi_ok"], f"Phi sum != 1 at u={u_grid_raw[i]:.3f}"
        assert val["psi_ok"], f"Psi sum != 1 at u={u_grid_raw[i]:.3f}"

    # Effective tail domain (atom case)
    q_u_star = q_s(u_star)
    tail_domain = compute_effective_tail_domain(cdf_s, q_u_star, u_star, float(s_max))

    return {
        "scenario": "Poisson_Discrete",
        "lambda1": lambda1, "lambda2": lambda2, "lam_s": lam_s,
        "u_star": u_star, "q_u_star": q_u_star,
        "tail_domain": tail_domain,
        "theta_nontrivial_vals": nontrivial_theta[:5],
        "nontrivial_theta_count": len(nontrivial_theta),
        "u_grid": u_grid_raw,
        "q_grid": q_grid, "es_grid": es_grid, "W_grid": W_grid,
        "phi1": phi1_arr, "psi1": psi1_arr,
        "phi2": phi2_arr, "psi2": psi2_arr,
        "g1_at_s": r1_at_s,
    }


# -------------------------------------------------------------------
# Scenario C: Constant conditional share (equality case)
# X_i = a_i * Z where Z ~ N(1, 1), so X_1/S = a_1/(a_1+a_2) = const
# -------------------------------------------------------------------

def run_equality_scenario(
    a1: float = 0.4, a2: float = 0.6,
    u_star: float = 0.6,
    n_mc: int = 50_000,
    rng_seed: int = 99,
) -> dict:
    """Run equality case: g_i constant => Phi = Psi."""

    rng = np.random.default_rng(rng_seed)
    a_s = a1 + a2
    Z = rng.normal(1.0, 1.0, n_mc)
    X1 = a1 * Z
    X2 = a2 * Z
    S = X1 + X2
    assert np.allclose(S, a_s * Z)

    mu_s = a_s  # E[Z] = 1
    sig_s = a_s  # std[Z] = 1

    def cdf_s(x: float) -> float:
        return float(stats.norm.cdf(x, loc=mu_s, scale=sig_s))

    u_grid = np.linspace(u_star, 0.999, 100)
    q_grid = np.array([stats.norm.ppf(u, loc=mu_s, scale=sig_s) for u in u_grid])
    es_grid = compute_es(q_grid, u_grid)

    R1 = compute_r_i(X1, S, c_i=a1 / a_s)

    phi1_list, psi1_list = [], []
    for q_u in q_grid:
        phi1_list.append(compute_phi_continuous(X1, S, q_u, 0.0))
        psi1_list.append(compute_psi_continuous(R1, S, q_u, 0.0))

    phi1_arr = np.array(phi1_list)
    psi1_arr = np.array(psi1_list)
    delta_arr = phi1_arr - psi1_arr
    equality_ok = np.nanmax(np.abs(delta_arr)) < 0.015  # MC noise tolerance

    return {
        "scenario": "Equality_Comonotonic",
        "a1": a1, "a2": a2, "expected_share": a1 / a_s,
        "equality_ok": equality_ok,
        "max_delta": float(np.nanmax(np.abs(delta_arr))),
        "phi1_mean": float(np.nanmean(phi1_arr)),
        "psi1_mean": float(np.nanmean(psi1_arr)),
        "u_grid": u_grid, "q_grid": q_grid, "es_grid": es_grid,
        "phi1": phi1_arr, "psi1": psi1_arr,
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_normal_scenario(res: dict, save_path: str) -> None:
    """Plot Normal scenario results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Exp 1A: Normal Scenario — Tail Allocation", fontsize=14)

    u = res["u_grid"]

    ax = axes[0, 0]
    ax.plot(u, res["q_grid"], label="q(u) = VaR_u[S]")
    ax.plot(u, res["es_grid"], label="ES_u(S)", linestyle="--")
    ax.set_xlabel("u"); ax.set_ylabel("Value"); ax.legend(); ax.set_title("Quantile & ES")

    ax = axes[0, 1]
    ax.plot(u, res["W_grid"], color="purple")
    ax.set_xlabel("u"); ax.set_ylabel("W_u"); ax.set_title("W_u = ES_u / q(u)")

    ax = axes[0, 2]
    ax.plot(u, res["phi1"], label="Φ₁,ᵤ", color="blue")
    ax.plot(u, res["psi1"], label="Ψ₁,ᵤ", color="red", linestyle="--")
    ax.plot(u, res["phi2"], label="Φ₂,ᵤ", color="cyan")
    ax.plot(u, res["psi2"], label="Ψ₂,ᵤ", color="orange", linestyle="--")
    ax.set_xlabel("u"); ax.set_ylabel("Allocation"); ax.legend(); ax.set_title("Phi & Psi")

    ax = axes[1, 0]
    delta = res["phi1"] - res["psi1"]
    ax.plot(u, delta, color="green")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("u"); ax.set_ylabel("Φ - Ψ"); ax.set_title("Delta = Phi1 - Psi1")

    ax = axes[1, 1]
    # Dominance sign
    sign_str = "Phi > Psi" if res["expected_sign"] > 0 else "Phi < Psi"
    ax.text(0.5, 0.5, f"Dominance: {sign_str}\nHolds: {res['dominance_holds']}\nEquality (mu_eq): {res['equality_holds']}",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("Dominance Check")
    ax.axis("off")

    ax = axes[1, 2]
    ax.text(0.5, 0.7,
            f"Tail domain lower open: {res['tail_domain']['lower_open']}\n"
            f"q(u*)={res['q_u_star']:.3f}\n"
            f"Localization valid (q(u*)>0): {res['q_u_star'] > 0}\n"
            f"theta_u max: {res['theta_u_max']:.2e}\n"
            f"Cov identity mean err: {res['cov_identity_mean_err']:.4f}",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Diagnostics")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_poisson_scenario(res: dict, save_path: str) -> None:
    """Plot Poisson scenario results."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Exp 1B: Poisson Scenario — Discrete Atom Handling", fontsize=13)

    u = res["u_grid"]

    axes[0].plot(u, res["q_grid"], label="q(u)", marker="o", markersize=2)
    axes[0].plot(u, res["es_grid"], label="ES_u(S)", linestyle="--")
    axes[0].set_xlabel("u"); axes[0].legend(); axes[0].set_title("Quantile & ES")

    axes[1].plot(u, res["phi1"], label="Φ₁,ᵤ", color="blue")
    axes[1].plot(u, res["psi1"], label="Ψ₁,ᵤ", color="red", linestyle="--")
    axes[1].set_xlabel("u"); axes[1].legend(); axes[1].set_title("Phi & Psi (Component 1)")

    delta = res["phi1"] - res["psi1"]
    axes[2].plot(u, delta, color="green")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlabel("u"); axes[2].set_ylabel("Φ - Ψ"); axes[2].set_title("Delta")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_equality_scenario(res: dict, save_path: str) -> None:
    """Plot equality scenario results."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Exp 1C: Equality Case — Constant g_i => Phi = Psi", fontsize=13)

    u = res["u_grid"]

    axes[0].plot(u, res["phi1"], label="Φ₁,ᵤ", color="blue", lw=2)
    axes[0].plot(u, res["psi1"], label="Ψ₁,ᵤ", color="red", linestyle="--", lw=2)
    axes[0].axhline(res["expected_share"], color="green", lw=1, linestyle=":", label=f"Expected r₁={res['expected_share']:.2f}")
    axes[0].set_xlabel("u"); axes[0].legend(); axes[0].set_title("Phi = Psi (equality)")

    delta = res["phi1"] - res["psi1"]
    axes[1].plot(u, delta, color="purple")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("u"); axes[1].set_ylabel("Φ - Ψ"); axes[1].set_title(f"Delta (max={res['max_delta']:.4f})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_experiment_1() -> pd.DataFrame:
    """Run all Experiment 1 scenarios and return summary DataFrame."""

    print("=" * 60)
    print("EXPERIMENT 1: General Tail-Allocation Framework Validation")
    print("=" * 60)

    # Scenario A: Normal continuous
    print("\n[1A] Normal continuous scenario...")
    res_norm = run_normal_scenario(mu1=0.5, sig1=1.0, mu2=1.5, sig2=1.2, u_star=0.5)
    plot_normal_scenario(res_norm, os.path.join(RESULTS_DIR, "exp1_normal_scenario.png"))
    print(f"  Tail domain lower_open={res_norm['tail_domain']['lower_open']}")
    print(f"  Equality (constant g): holds={res_norm['equality_holds']}, max_err={res_norm['equality_max_err']:.4f}")
    print(f"  Dominance sign={res_norm['expected_sign']}: holds={res_norm['dominance_holds']}")
    print(f"  Cov identity mean_err={res_norm['cov_identity_mean_err']:.4f}")

    # Scenario B: Poisson discrete
    print("\n[1B] Poisson discrete scenario...")
    res_pois = run_poisson_scenario(lambda1=3.0, lambda2=5.0, u_star=0.5)
    plot_poisson_scenario(res_pois, os.path.join(RESULTS_DIR, "exp1_poisson_scenario.png"))
    print(f"  Tail domain lower_open={res_pois['tail_domain']['lower_open']}")
    print(f"  Nontrivial theta_u count={res_pois['nontrivial_theta_count']}")
    print(f"  Sample theta values: {[f'{t:.4f}' for t in res_pois['theta_nontrivial_vals']]}")

    # Scenario C: Equality
    print("\n[1C] Equality scenario (constant g_i)...")
    res_eq = run_equality_scenario(a1=0.4, a2=0.6, u_star=0.6)
    plot_equality_scenario(res_eq, os.path.join(RESULTS_DIR, "exp1_equality_scenario.png"))
    print(f"  Equality holds: {res_eq['equality_ok']}, max_delta={res_eq['max_delta']:.4f}")

    # Scenario D: Increasing g_1 (mu1 < beta1*mu_S => Phi > Psi)
    print("\n[1D] Increasing g_i (Phi > Psi)...")
    res_inc = run_normal_scenario(mu1=0.3, sig1=2.0, mu2=1.5, sig2=0.5, u_star=0.7)
    print(f"  Dominance (mu1={res_inc['mu1']}, beta1*mu_S={res_inc['beta1']*res_inc['mu_s']:.3f}): "
          f"holds={res_inc['dominance_holds']}, expected_sign={res_inc['expected_sign']}")

    # Scenario E: Decreasing g_1 (mu1 > beta1*mu_S => Phi < Psi)
    print("\n[1E] Decreasing g_i (Phi < Psi)...")
    res_dec = run_normal_scenario(mu1=3.0, sig1=0.5, mu2=0.5, sig2=2.0, u_star=0.6)
    print(f"  Dominance (mu1={res_dec['mu1']}, beta1*mu_S={res_dec['beta1']*res_dec['mu_s']:.3f}): "
          f"holds={res_dec['dominance_holds']}, expected_sign={res_dec['expected_sign']}")

    # Summary table
    summary = pd.DataFrame([
        {
            "Scenario": "Normal continuous",
            "Equality_holds": res_norm["equality_holds"],
            "Dominance_holds": res_norm["dominance_holds"],
            "Cov_identity_err": res_norm["cov_identity_mean_err"],
            "theta_u_max": res_norm["theta_u_max"],
            "tail_domain_open": res_norm["tail_domain"]["lower_open"],
        },
        {
            "Scenario": "Poisson discrete",
            "Equality_holds": "N/A",
            "Dominance_holds": "N/A",
            "Cov_identity_err": "N/A",
            "theta_u_max": max(res_pois["theta_nontrivial_vals"]) if res_pois["theta_nontrivial_vals"] else 0.0,
            "tail_domain_open": res_pois["tail_domain"]["lower_open"],
        },
        {
            "Scenario": "Equality (const g_i)",
            "Equality_holds": res_eq["equality_ok"],
            "Dominance_holds": "N/A",
            "Cov_identity_err": "N/A",
            "theta_u_max": 0.0,
            "tail_domain_open": True,
        },
        {
            "Scenario": "Increasing g_i",
            "Equality_holds": False,
            "Dominance_holds": res_inc["dominance_holds"],
            "Cov_identity_err": res_inc["cov_identity_mean_err"],
            "theta_u_max": 0.0,
            "tail_domain_open": True,
        },
        {
            "Scenario": "Decreasing g_i",
            "Equality_holds": False,
            "Dominance_holds": res_dec["dominance_holds"],
            "Cov_identity_err": res_dec["cov_identity_mean_err"],
            "theta_u_max": 0.0,
            "tail_domain_open": True,
        },
    ])

    summary.to_csv(os.path.join(RESULTS_DIR, "exp1_summary.csv"), index=False)
    print("\nSummary saved to results/exp1_summary.csv")
    return summary


if __name__ == "__main__":
    summary = run_experiment_1()
    print("\n" + summary.to_string(index=False))
