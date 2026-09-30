"""
Experiment 2: Independent Additive EDM Experiments

Validates:
- Homogeneous parameter EDMs: Phi = Psi = lambda_i / sum(lambda_j)
- Heterogeneous scaled Poisson: unique global equality
- Heterogeneous non-scaled-Poisson: failure of equality
- Normal example: exact g_i, dominance via mean-variance condition
- Gamma n=2 example: parameter ordering and dominance
- Tweedie monotonicity proxy via iota(theta)
- Generalized hyperbolic secant localization threshold
- Saddle-point proxy vs exact allocations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.edm_families import (
    NormalEDM, PoissonEDM, ScaledPoissonEDM, GammaEDM,
    TweedieEDM, GenHypSecantEDM,
    compute_saddle_point, compute_g_hat_i, compute_aggregate_mean,
)
from src.tail_allocation import (
    compute_quantile_grid, compute_es, compute_W_u,
    compute_phi_continuous, compute_psi_continuous, compute_r_i,
    validate_allocations,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_MC = 20_000
RNG = np.random.default_rng(2024)


# -------------------------------------------------------------------
# Utility: run MC allocation over a u-grid
# -------------------------------------------------------------------

def mc_allocations(
    x_samples: np.ndarray,  # shape (n_mc, n_comp)
    s_samples: np.ndarray,  # shape (n_mc,)
    u_grid: np.ndarray,
    q_grid: np.ndarray,
    c_i: float = 0.0,
) -> tuple:
    """Compute Phi and Psi via fully-vectorised MC over a u-grid.

    For continuous distributions theta_u = 0, so pi_u(s) = 1_{s > q_u}.
    All n_comp × n_u pairs are evaluated in a single set of matrix operations.

    Parameters
    ----------
    x_samples : (n_mc, n_comp)
    s_samples : (n_mc,)
    u_grid    : (n_u,)
    q_grid    : (n_u,)  — quantile values matching u_grid
    c_i       : fallback share when S = 0

    Returns
    -------
    phi_arr : (n_comp, n_u)
    psi_arr : (n_comp, n_u)
    """
    n_mc, n_comp = x_samples.shape
    n_u = len(u_grid)

    # R_i = X_i / S; shape (n_mc, n_comp)
    r_mat = np.column_stack(
        [compute_r_i(x_samples[:, i], s_samples, c_i=c_i) for i in range(n_comp)]
    )

    # Tail indicator matrix: shape (n_mc, n_u)  — float32 saves memory
    # mask[i, j] = 1 if s_samples[i] > q_grid[j]  (theta_u = 0 for continuous)
    mask = (s_samples[:, None] > q_grid[None, :]).astype(np.float32)  # (n_mc, n_u)

    # tail_prob[j] = E[pi_u(S)]  shape (n_u,)
    tail_prob = mask.mean(axis=0)

    # s_tail[j] = E[S * pi_u(S)]  shape (n_u,)
    s_tail = (s_samples.astype(np.float32) @ mask) / n_mc

    # x_tail[i, j] = E[X_i * pi_u(S)]  shape (n_comp, n_u)
    # matmul: (n_comp, n_mc) @ (n_mc, n_u) → (n_comp, n_u)
    x_tail = (x_samples.T.astype(np.float32) @ mask) / n_mc

    # r_tail[i, j] = E[R_i * pi_u(S)]  shape (n_comp, n_u)
    r_tail = (r_mat.T.astype(np.float32) @ mask) / n_mc

    # Avoid division by zero
    s_tail_safe = np.where(np.abs(s_tail) < 1e-15, np.nan, s_tail)
    tail_prob_safe = np.where(tail_prob < 1e-15, np.nan, tail_prob)

    phi_arr = x_tail / s_tail_safe          # (n_comp, n_u)
    psi_arr = r_tail / tail_prob_safe        # (n_comp, n_u)

    return phi_arr, psi_arr


# -------------------------------------------------------------------
# Scenario 1: Homogeneous-parameter Normal EDMs
# -------------------------------------------------------------------

def run_homogeneous_normal(
    mus: list = [1.0, 2.0, 1.5],
    sigma: float = 1.0,
    u_star: float = 0.6,
) -> dict:
    """Homogeneous Normal: theta_i = mu_i/sigma_i^2 all equal => Phi = Psi = lam_i/sum."""

    family = NormalEDM()
    mus = np.array(mus)
    n = len(mus)
    sigmas = np.full(n, sigma)
    lambdas = sigmas**2  # lambda_i = sigma_i^2

    # All theta_i = mu_i / sigma_i^2 must be equal -> set equal means
    theta_common = 1.0
    mus_eq = theta_common * sigmas**2  # mu_i = theta * lambda_i
    mus_eq_arr = mus_eq

    mu_s = mus_eq_arr.sum()
    sig_s = np.sqrt((sigmas**2).sum())
    expected_share = lambdas / lambdas.sum()

    samples = family.sample_components(lambdas, mus_eq_arr / lambdas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)

    u_grid = np.linspace(u_star, 0.999, 80)
    q_grid = np.array([stats.norm.ppf(u, loc=mu_s, scale=sig_s) for u in u_grid])

    phi_arr, psi_arr = mc_allocations(samples.T, s, u_grid, q_grid)

    # Check equality at each u
    errors = np.abs(phi_arr - psi_arr)
    equality_ok = errors[:, :].max() < 0.03  # MC tolerance

    # Check allocation = expected_share
    share_errors = np.abs(phi_arr.T - expected_share).max()

    return {
        "scenario": "Homogeneous_Normal",
        "expected_shares": expected_share,
        "phi_mean": phi_arr.mean(axis=1),
        "psi_mean": psi_arr.mean(axis=1),
        "equality_ok": equality_ok,
        "max_error": float(errors.max()),
        "share_error": float(share_errors),
        "u_grid": u_grid, "phi": phi_arr, "psi": psi_arr,
    }


# -------------------------------------------------------------------
# Scenario 2: Heterogeneous scaled Poisson (unique global equality)
# -------------------------------------------------------------------

def run_heterogeneous_scaled_poisson(
    lambdas: list = [2.0, 3.0],
    thetas: list = [0.5, 1.0],
    a: float = 1.0, b: float = 1.0,  # standard Poisson: a=b=1
    u_star: float = 0.4,
) -> dict:
    """Scaled Poisson with heterogeneous theta_i: global equality holds."""

    family = ScaledPoissonEDM(a=a, b=b)
    lambdas = np.array(lambdas)
    thetas = np.array(thetas)

    # Expected shares for scaled Poisson
    r_i = lambdas * np.exp(b * thetas) / np.sum(lambdas * np.exp(b * thetas))

    samples = family.sample_components(lambdas, thetas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)

    mu_s = compute_aggregate_mean(family, lambdas, thetas)
    sig_s = np.std(s)

    def cdf_s(x: float) -> float:
        return float(np.mean(s <= x))

    u_grid = np.linspace(u_star, 0.99, 60)
    q_grid = np.percentile(s, u_grid * 100)

    phi_arr, psi_arr = mc_allocations(samples.T, s, u_grid, q_grid)
    errors = np.abs(phi_arr.T - r_i)
    phi_mean_err = float(errors.max())
    equality_ok = phi_mean_err < 0.05

    return {
        "scenario": "Heterogeneous_ScaledPoisson",
        "lambdas": lambdas, "thetas": thetas,
        "expected_shares": r_i,
        "phi_mean": phi_arr.mean(axis=1),
        "psi_mean": psi_arr.mean(axis=1),
        "equality_ok": equality_ok,
        "max_error_from_r_i": phi_mean_err,
        "u_grid": u_grid, "phi": phi_arr, "psi": psi_arr,
    }


# -------------------------------------------------------------------
# Scenario 3: Heterogeneous Gamma (non-equality)
# -------------------------------------------------------------------

def run_heterogeneous_gamma(
    lambdas: list = [2.0, 3.0],
    thetas: list = [-0.5, -1.0],
    u_star: float = 0.5,
) -> dict:
    """Heterogeneous Gamma EDM: no global equality expected."""

    family = GammaEDM()
    lambdas = np.array(lambdas)
    thetas = np.array(thetas)

    samples = family.sample_components(lambdas, thetas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)
    pos_mask = s > 0          # Gamma is positive
    s = s[pos_mask]
    samples = samples[:, pos_mask]  # (n_comp, n_pos)

    u_grid = np.linspace(u_star, 0.99, 60)
    q_grid = np.percentile(s, u_grid * 100)

    n_mc_use = min(N_MC, s.shape[0])
    phi_arr, psi_arr = mc_allocations(samples[:, :n_mc_use].T, s[:n_mc_use], u_grid, q_grid)

    delta = phi_arr - psi_arr
    # Check that delta is NOT close to 0 everywhere (threshold 0.005)
    not_equal = float(np.abs(delta).mean()) > 0.005

    return {
        "scenario": "Heterogeneous_Gamma",
        "lambdas": lambdas, "thetas": thetas,
        "phi_mean": phi_arr.mean(axis=1),
        "psi_mean": psi_arr.mean(axis=1),
        "not_equal": not_equal,
        "mean_delta": float(np.abs(delta).mean()),
        "u_grid": u_grid, "phi": phi_arr, "psi": psi_arr,
    }


# -------------------------------------------------------------------
# Scenario 4: Normal exact conditional share validation
# -------------------------------------------------------------------

def run_normal_exact_g(
    mus: list = [1.0, 2.0],
    sigmas: list = [1.0, 1.5],
    u_star: float = 0.5,
) -> dict:
    """Normal example: verify exact g_i formula and dominance condition."""

    family = NormalEDM()
    mus = np.array(mus)
    sigmas = np.array(sigmas)
    lambdas = sigmas**2
    thetas = mus / sigmas**2

    mu_s = mus.sum()
    sig_s = np.sqrt((sigmas**2).sum())
    beta = sigmas**2 / sig_s**2

    # Exact g_i(s)
    def g1_exact(s: float) -> float:
        if abs(s) < 1e-12:
            return float(beta[0])
        return float(beta[0] + (mus[0] - beta[0] * mu_s) / s)

    # Dominance condition: g_i(s)=beta_i+(mu_i-beta_i*mu_S)/s on s>0.
    # dg_i/ds < 0 iff mu_i > beta_i*mu_S (cond1>0) => g_i decreasing => Phi_i < Psi_i
    # dg_i/ds > 0 iff mu_i < beta_i*mu_S (cond1<0) => g_i increasing => Phi_i > Psi_i
    cond1 = mus[0] - beta[0] * mu_s  # cond1<0 => Phi>Psi; cond1>0 => Phi<Psi
    cond2 = mus[1] - beta[1] * mu_s

    samples = family.sample_components(lambdas, thetas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)
    pos_mask = s > 0.1
    s_pos = s[pos_mask]
    samples_pos = samples[:, pos_mask]  # (n_comp, n_pos)

    u_grid = np.linspace(u_star, 0.999, 80)
    q_grid = np.array([stats.norm.ppf(u, loc=mu_s, scale=sig_s) for u in u_grid])

    n_use = min(N_MC, len(s_pos))
    phi_arr, psi_arr = mc_allocations(
        samples_pos[:, :n_use].T, s_pos[:n_use], u_grid, q_grid
    )

    delta = phi_arr[0] - psi_arr[0]
    # expected sign of delta = -sign(cond1): cond1<0 => increasing g => Phi>Psi
    sign_ok = (np.sign(np.nanmean(delta[len(delta) // 4:])) == -np.sign(cond1))

    # Saddle-point approximation
    s_test_vals = np.linspace(mu_s + 0.5 * sig_s, mu_s + 3 * sig_s, 20)
    g_hat_vals, g_exact_vals = [], []
    for s_test in s_test_vals:
        try:
            t_hat = compute_saddle_point(family, lambdas, thetas, s_test)
            g_hat = compute_g_hat_i(family, lambdas, thetas, s_test, 0)
            g_hat_vals.append(g_hat)
        except Exception:
            g_hat_vals.append(np.nan)
        g_exact_vals.append(g1_exact(s_test))

    sp_errors = [abs(gh - ge) for gh, ge in zip(g_hat_vals, g_exact_vals)
                 if not np.isnan(gh)]

    return {
        "scenario": "Normal_Exact_g",
        "mus": mus, "sigmas": sigmas, "beta": beta,
        "cond1": cond1, "cond2": cond2,
        "dominance_sign_ok": sign_ok,
        "sp_mean_error": float(np.mean(sp_errors)) if sp_errors else np.nan,
        "phi_mean": phi_arr.mean(axis=1),
        "psi_mean": psi_arr.mean(axis=1),
        "delta_mean": float(np.nanmean(delta)),
        "u_grid": u_grid, "phi": phi_arr, "psi": psi_arr,
    }


# -------------------------------------------------------------------
# Scenario 5: Gamma n=2 exact dominance
# -------------------------------------------------------------------

def run_gamma_n2(
    lambda1: float = 2.0, lambda2: float = 3.0,
    theta1: float = -1.0, theta2: float = -0.5,
    u_star: float = 0.5,
) -> dict:
    """Gamma n=2 example: rate ordering drives dominance."""

    family = GammaEDM()
    lambdas = np.array([lambda1, lambda2])
    thetas = np.array([theta1, theta2])

    # iota(theta) = -theta for Gamma
    iota1 = family.iota(theta1)  # = -theta1 > 0
    iota2 = family.iota(theta2)

    samples = family.sample_components(lambdas, thetas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)
    pos_mask = s > 0
    s_pos = s[pos_mask]
    samples_pos = samples[:, pos_mask]  # (n_comp, n_pos)

    u_grid = np.linspace(u_star, 0.99, 60)
    q_grid = np.percentile(s_pos, u_grid * 100)

    n_use = min(N_MC, len(s_pos))
    phi_arr, psi_arr = mc_allocations(samples_pos[:, :n_use].T, s_pos[:n_use], u_grid, q_grid)
    delta = phi_arr[0] - psi_arr[0]

    # If theta1 > theta2 (less negative), iota1 < iota2 => ...
    # The paper's dominance for Gamma: ordering on rate -theta_i
    rate1, rate2 = -theta1, -theta2
    expected_sign = 1 if rate1 < rate2 else -1  # lower rate => larger scale => heavier

    return {
        "scenario": "Gamma_n2",
        "lambda1": lambda1, "lambda2": lambda2,
        "theta1": theta1, "theta2": theta2,
        "iota1": iota1, "iota2": iota2,
        "rate1": rate1, "rate2": rate2,
        "mean_delta": float(np.nanmean(delta)),
        "sign_consistent": (np.sign(np.nanmean(delta)) == expected_sign),
        "u_grid": u_grid, "phi": phi_arr, "psi": psi_arr,
    }


# -------------------------------------------------------------------
# Scenario 6: Tweedie monotonicity proxy
# -------------------------------------------------------------------

def run_tweedie_proxy(
    p: float = 1.5,
    lambdas: list = [2.0, 3.0],
    thetas: list = [-1.0, -0.5],
    u_star: float = 0.5,
) -> dict:
    """Tweedie example: saddle-point proxy and iota ordering."""

    family = TweedieEDM(p=p)
    lambdas = np.array(lambdas)
    thetas = np.array(thetas)

    iota_vals = [family.iota(t) for t in thetas]
    # iota(theta) = (1-p)*theta for general Tweedie

    samples = family.sample_components(lambdas, thetas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)
    pos_mask = s > 1e-8
    s_pos = s[pos_mask]
    samples_pos = samples[:, pos_mask]  # (n_comp, n_pos)

    u_grid = np.linspace(u_star, 0.99, 50)
    q_grid = np.percentile(s_pos, u_grid * 100)

    n_use = min(N_MC, len(s_pos))
    phi_arr, psi_arr = mc_allocations(samples_pos[:, :n_use].T, s_pos[:n_use], u_grid, q_grid)

    # Saddle-point proxy g_hat at various s values
    s_test = np.linspace(q_grid[5], q_grid[-5], 15)
    g_hat_list = []
    for s_val in s_test:
        try:
            g_hat = compute_g_hat_i(family, lambdas, thetas, s_val, 0)
            g_hat_list.append(g_hat)
        except Exception:
            g_hat_list.append(np.nan)

    return {
        "scenario": f"Tweedie_p{p}",
        "p": p, "lambdas": lambdas, "thetas": thetas,
        "iota_vals": iota_vals,
        "phi_mean": phi_arr.mean(axis=1),
        "psi_mean": psi_arr.mean(axis=1),
        "g_hat_sample": g_hat_list,
        "u_grid": u_grid, "phi": phi_arr, "psi": psi_arr,
    }


# -------------------------------------------------------------------
# Scenario 7: Generalized Hyperbolic Secant localization
# -------------------------------------------------------------------

def run_ghs_localization(
    lambdas: list = [1.0, 1.0],
    thetas: list = [0.2, 0.6],
    u_star: float = 0.7,
) -> dict:
    """GHS scenario: localization threshold t* = pi/4 - theta_min."""

    family = GenHypSecantEDM()
    lambdas = np.array(lambdas)
    thetas = np.array(thetas)

    theta_min = thetas.min()
    theta_max = thetas.max()
    assert theta_max - theta_min < np.pi / 4, "Condition theta_max - theta_min < pi/4 required"

    t_star = family.switching_tilt(thetas)
    iota_vals = [family.iota(t) for t in thetas]

    samples = family.sample_components(lambdas, thetas, N_MC, RNG)
    s = samples.sum(axis=0)  # (N_MC,)

    u_grid_global = np.linspace(0.3, 0.99, 50)
    u_grid_local = np.linspace(u_star, 0.99, 50)

    q_global = np.percentile(s, u_grid_global * 100)
    q_local = np.percentile(s, u_grid_local * 100)

    n_use = N_MC
    phi_global, psi_global = mc_allocations(samples[:, :n_use].T, s[:n_use], u_grid_global, q_global)
    phi_local, psi_local = mc_allocations(samples[:, :n_use].T, s[:n_use], u_grid_local, q_local)

    delta_global = phi_global[0] - psi_global[0]
    delta_local = phi_local[0] - psi_local[0]

    # Localized monotonicity should be more consistent
    sign_local = np.sign(np.nanmean(delta_local))

    return {
        "scenario": "GHS_Localization",
        "lambdas": lambdas, "thetas": thetas,
        "theta_min": theta_min, "theta_max": theta_max,
        "t_star": t_star,
        "iota_vals": iota_vals,
        "sign_local": sign_local,
        "mean_delta_global": float(np.nanmean(np.abs(delta_global))),
        "mean_delta_local": float(np.nanmean(np.abs(delta_local))),
        "u_grid_global": u_grid_global,
        "u_grid_local": u_grid_local,
        "phi_global": phi_global, "psi_global": psi_global,
        "phi_local": phi_local, "psi_local": psi_local,
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_edm_comparison(results: list, save_path: str) -> None:
    """Plot Phi, Psi, Delta for multiple EDM scenarios."""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Exp 2: Independent Additive EDM Experiments", fontsize=14)

    scenarios_to_plot = [r for r in results if "u_grid" in r and "phi" in r][:9]
    for idx, res in enumerate(scenarios_to_plot):
        ax = axes[idx // 3, idx % 3]
        u = res["u_grid"]
        phi0 = res["phi"][0]
        psi0 = res["psi"][0]
        ax.plot(u, phi0, label="Φ₁", color="blue", lw=1.5)
        ax.plot(u, psi0, label="Ψ₁", color="red", linestyle="--", lw=1.5)
        ax.set_title(res["scenario"], fontsize=9)
        ax.set_xlabel("u", fontsize=8)
        ax.legend(fontsize=7)
        delta = phi0 - psi0
        color = "green" if np.nanmean(delta) > 0 else "orange"
        ax2 = ax.twinx()
        ax2.plot(u, delta, color=color, lw=0.8, linestyle=":", alpha=0.7)
        ax2.set_ylabel("Δ", fontsize=7, color=color)
        ax2.tick_params(axis="y", colors=color, labelsize=6)

    for idx in range(len(scenarios_to_plot), 9):
        axes[idx // 3, idx % 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_experiment_2() -> pd.DataFrame:
    """Run all Experiment 2 scenarios."""

    print("=" * 60)
    print("EXPERIMENT 2: Independent Additive EDM Experiments")
    print("=" * 60)

    results = []

    print("\n[2.1] Homogeneous Normal (equality expected)...")
    res1 = run_homogeneous_normal()
    results.append(res1)
    print(f"  Equality holds: {res1['equality_ok']}, max_err={res1['max_error']:.4f}")

    print("\n[2.2] Heterogeneous Scaled Poisson (unique global equality)...")
    res2 = run_heterogeneous_scaled_poisson()
    results.append(res2)
    print(f"  Equality holds: {res2['equality_ok']}, max_err from r_i={res2['max_error_from_r_i']:.4f}")

    print("\n[2.3] Heterogeneous Gamma (non-equality expected)...")
    res3 = run_heterogeneous_gamma()
    results.append(res3)
    print(f"  Not equal: {res3['not_equal']}, mean_delta={res3['mean_delta']:.4f}")

    print("\n[2.4] Normal exact g_i validation...")
    res4 = run_normal_exact_g(mus=[2.0, 1.0], sigmas=[1.0, 1.5])
    results.append(res4)
    print(f"  Dominance sign OK: {res4['dominance_sign_ok']}, SP error={res4['sp_mean_error']:.6f}")

    print("\n[2.5] Gamma n=2 dominance...")
    res5 = run_gamma_n2(lambda1=2.0, lambda2=3.0, theta1=-1.0, theta2=-0.5)
    results.append(res5)
    print(f"  Sign consistent: {res5['sign_consistent']}, mean_delta={res5['mean_delta']:.4f}")

    print("\n[2.6] Tweedie proxy (p=1.5)...")
    res6 = run_tweedie_proxy(p=1.5)
    results.append(res6)
    print(f"  iota values: {[f'{v:.3f}' for v in res6['iota_vals']]}")

    print("\n[2.7] GHS localization...")
    res7 = run_ghs_localization()
    results.append(res7)
    print(f"  t_star={res7['t_star']:.4f}, sign_local={res7['sign_local']}")

    plot_edm_comparison(results, os.path.join(RESULTS_DIR, "exp2_edm_comparison.png"))

    # Build summary
    summary = pd.DataFrame([
        {
            "Scenario": res1["scenario"],
            "Equality_holds": res1["equality_ok"],
            "Max_delta": res1["max_error"],
            "Note": f"shares={[f'{s:.2f}' for s in res1['expected_shares']]}",
        },
        {
            "Scenario": res2["scenario"],
            "Equality_holds": res2["equality_ok"],
            "Max_delta": res2["max_error_from_r_i"],
            "Note": f"r_i={[f'{s:.3f}' for s in res2['expected_shares']]}",
        },
        {
            "Scenario": res3["scenario"],
            "Equality_holds": not res3["not_equal"],
            "Max_delta": res3["mean_delta"],
            "Note": "Heterogeneous Gamma: no global equality",
        },
        {
            "Scenario": res4["scenario"],
            "Equality_holds": False,
            "Max_delta": abs(res4["delta_mean"]),
            "Note": f"cond1={res4['cond1']:.3f}, dom_sign_ok={res4['dominance_sign_ok']}",
        },
        {
            "Scenario": res5["scenario"],
            "Equality_holds": False,
            "Max_delta": abs(res5["mean_delta"]),
            "Note": f"sign_ok={res5['sign_consistent']}",
        },
        {
            "Scenario": res6["scenario"],
            "Equality_holds": False,
            "Max_delta": float(np.nanmean(np.abs(res6["phi"][0] - res6["psi"][0]))),
            "Note": f"p={res6['p']}, iota={[f'{v:.3f}' for v in res6['iota_vals']]}",
        },
        {
            "Scenario": res7["scenario"],
            "Equality_holds": False,
            "Max_delta": res7["mean_delta_local"],
            "Note": f"t_star={res7['t_star']:.4f}, localized sign={res7['sign_local']}",
        },
    ])

    summary.to_csv(os.path.join(RESULTS_DIR, "exp2_summary.csv"), index=False)
    print("\nSummary saved to results/exp2_summary.csv")
    return summary


if __name__ == "__main__":
    summary = run_experiment_2()
    print("\n" + summary.to_string(index=False))
