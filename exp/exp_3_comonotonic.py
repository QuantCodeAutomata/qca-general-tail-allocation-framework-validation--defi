"""
Experiment 3: Comonotonic Risk Experiments

Validates:
- Comonotonic quantile identity q_S(u) = sum_i q_{X_i}(u)
- Equality theorem: Phi = Psi iff q_{X_i}(u) proportional to q_S(u)
- Dominance theorem: sign of Phi - Psi tracks monotonicity of h(u) = q_Xi/q_S
- Location-scale families: closed-form discrepancy via ES and HS
- Pareto marginals: equality for equal alphas, dominance/concentration for unequal alphas
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

from src.comonotonic import (
    compute_comonotonic_aggregate_quantile_grid,
    compute_phi_comonotonic,
    compute_psi_comonotonic,
    compute_h_ratio,
    compute_es_comonotonic,
    compute_hs_u,
    compute_es_hs_on_grid,
    compute_location_scale_discrepancy,
    compute_pareto_quantile,
    compute_pareto_asymptotic_allocation,
    compute_allocations_on_grid,
    check_proportional_quantile_equality,
    check_ratio_monotonicity,
    build_pareto_scenario,
    LocationScaleScenario,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_POINTS = 1200
U_MAX = 1.0 - 1e-7


# -------------------------------------------------------------------
# Scenario 1: Proportional quantile equality
# q_{X_i}(u) = a_i * f(u) => h(u) = a_i / sum(a_j) = const => Phi = Psi
# -------------------------------------------------------------------

def run_proportional_quantile_equality(
    a: list = [0.3, 0.7],
    u_star: float = 0.3,
    n_u: int = 100,
) -> dict:
    """Proportional quantile case: exact equality Phi = Psi."""

    a = np.array(a)
    a_s = a.sum()
    expected_shares = a / a_s

    # Common shape f(u) = Normal ppf (baseline)
    from scipy.stats import norm

    def f(u: float) -> float:
        return float(norm.ppf(u, loc=2.0, scale=1.0))  # shifted normal, positive on tail

    q_xi_funcs = [lambda u, ai=ai: ai * f(u) for ai in a]
    q_s_func = lambda u: a_s * f(u)

    u_grid = np.linspace(u_star, 0.995, n_u)

    # Check positivity on tail
    q_s_vals = np.array([q_s_func(u) for u in u_grid])
    u_pos_start = u_grid[q_s_vals > 0][0] if (q_s_vals > 0).any() else u_star
    u_grid = np.linspace(max(u_pos_start, u_star), 0.995, n_u)

    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi = result["phi"]  # shape (n_comp, n_u)
    psi = result["psi"]
    h = result["h"]
    delta = result["delta"]

    # Check h constant
    for i in range(len(a)):
        h_i = h[i]
        valid = h_i[~np.isnan(h_i)]
        h_variation = float(np.ptp(valid)) if len(valid) > 1 else 0.0
        assert h_variation < 0.01, f"h not constant for i={i}: variation={h_variation:.4f}"

    equality_ok = bool(np.nanmax(np.abs(delta)) < 1e-4)

    # Check shares match expected
    phi_errors = np.abs(phi.T - expected_shares)
    psi_errors = np.abs(psi.T - expected_shares)

    return {
        "scenario": "Proportional_Quantile_Equality",
        "a": a, "expected_shares": expected_shares,
        "equality_ok": equality_ok,
        "max_delta": float(np.nanmax(np.abs(delta))),
        "phi_share_err": float(phi_errors.max()),
        "psi_share_err": float(psi_errors.max()),
        "u_grid": u_grid,
        "phi": phi, "psi": psi, "h": h, "delta": delta,
        "q_s": result["q_s"],
    }


# -------------------------------------------------------------------
# Scenario 2a: Increasing ratio h(u) => Phi > Psi
# -------------------------------------------------------------------

def run_increasing_ratio(
    u_star: float = 0.3,
    n_u: int = 100,
) -> dict:
    """Increasing h(u) = q_Xi(u)/q_S(u) => Phi > Psi."""

    # X_1 with heavier tail: Pareto(alpha=2, sigma=1)
    # X_2 lighter: Pareto(alpha=4, sigma=1)
    # h_1(u) = q_X1(u)/q_S(u) = (1-u)^{-1/2} / [(1-u)^{-1/2} + (1-u)^{-1/4}]
    #        = 1 / [1 + (1-u)^{1/4}] which is increasing as u -> 1

    def q_x1(u: float) -> float:
        return compute_pareto_quantile(2.0, 1.0, u)

    def q_x2(u: float) -> float:
        return compute_pareto_quantile(4.0, 1.0, u)

    q_xi_funcs = [q_x1, q_x2]

    u_grid = np.linspace(u_star, 0.995, n_u)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi, psi, h, delta = result["phi"], result["psi"], result["h"], result["delta"]

    # h_1 should be increasing
    mon = check_ratio_monotonicity(h[0], u_grid)
    increasing = mon["monotone_increasing"]

    # Phi[0] > Psi[0] for increasing h
    phi1_gt_psi1 = float(np.nanmean(delta[0])) > 0

    return {
        "scenario": "Increasing_Ratio_h",
        "h_monotone_increasing": increasing,
        "phi1_gt_psi1": phi1_gt_psi1,
        "mean_delta": float(np.nanmean(delta[0])),
        "u_grid": u_grid,
        "phi": phi, "psi": psi, "h": h, "delta": delta,
        "q_s": result["q_s"],
    }


# -------------------------------------------------------------------
# Scenario 2b: Decreasing ratio h(u) => Phi < Psi
# -------------------------------------------------------------------

def run_decreasing_ratio(
    u_star: float = 0.3,
    n_u: int = 100,
) -> dict:
    """Decreasing h(u) => Phi < Psi for component 1."""

    # Now X_1 lighter tail, X_2 heavier
    def q_x1(u: float) -> float:
        return compute_pareto_quantile(4.0, 1.0, u)  # lighter

    def q_x2(u: float) -> float:
        return compute_pareto_quantile(2.0, 1.0, u)  # heavier

    q_xi_funcs = [q_x1, q_x2]

    u_grid = np.linspace(u_star, 0.995, n_u)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi, psi, h, delta = result["phi"], result["psi"], result["h"], result["delta"]

    mon = check_ratio_monotonicity(h[0], u_grid)
    decreasing = mon["monotone_decreasing"]

    phi1_lt_psi1 = float(np.nanmean(delta[0])) < 0

    return {
        "scenario": "Decreasing_Ratio_h",
        "h_monotone_decreasing": decreasing,
        "phi1_lt_psi1": phi1_lt_psi1,
        "mean_delta": float(np.nanmean(delta[0])),
        "u_grid": u_grid,
        "phi": phi, "psi": psi, "h": h, "delta": delta,
        "q_s": result["q_s"],
    }


# -------------------------------------------------------------------
# Scenario 3: Location-scale family
# -------------------------------------------------------------------

def run_location_scale(
    mus: list = [1.0, 2.0],
    sigmas: list = [1.0, 2.0],
    u_star: float = 0.6,
    n_u: int = 100,
) -> dict:
    """Location-scale: verify discrepancy formula Phi - Psi = (sigma_i*mu_S - sigma_S*mu_i)/sigma_S * (1/HS - 1/ES)."""

    mus = np.array(mus, dtype=float)
    sigmas = np.array(sigmas, dtype=float)
    from scipy.stats import logistic

    # Baseline quantile z(u) = standard logistic ppf
    z_func = lambda u: float(logistic.ppf(u))

    scenario = LocationScaleScenario(mus=mus, sigmas=sigmas, z_func=z_func)
    q_xi_funcs = [lambda u, i=i: scenario.q_xi(i, u) for i in range(len(mus))]

    # Ensure positive on tail
    u_min_pos = u_star
    q_s_at_u = scenario.q_s(u_min_pos)
    while q_s_at_u <= 0 and u_min_pos < 0.95:
        u_min_pos += 0.05
        q_s_at_u = scenario.q_s(u_min_pos)

    u_grid = np.linspace(u_min_pos, 0.995, n_u)

    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi, psi, h, delta = result["phi"], result["psi"], result["h"], result["delta"]

    # Compare to closed-form discrepancy — batch ES/HS computation
    es_grid, hs_grid = compute_es_hs_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    closed_forms = np.array([
        compute_location_scale_discrepancy(
            mus[0], sigmas[0], scenario.mu_s, scenario.sigma_s, es_grid[j], hs_grid[j]
        ) if np.isfinite(es_grid[j]) and np.isfinite(hs_grid[j]) else np.nan
        for j in range(len(u_grid))
    ])
    discrepancy_err = np.where(
        np.isfinite(closed_forms),
        np.abs(delta[0] - closed_forms),
        np.nan,
    ).tolist()

    # Equality condition: mu_i/sigma_i = mu_S/sigma_S
    equality_ratio = mus / sigmas
    agg_ratio = scenario.mu_s / scenario.sigma_s
    equality_condition = [abs(r - agg_ratio) < 1e-10 for r in equality_ratio]

    return {
        "scenario": "Location_Scale",
        "mus": mus, "sigmas": sigmas,
        "mu_s": scenario.mu_s, "sigma_s": scenario.sigma_s,
        "equality_condition": equality_condition,
        "discrepancy_mean_err": float(np.nanmean(discrepancy_err)),
        "discrepancy_max_err": float(np.nanmax(np.abs(discrepancy_err))) if discrepancy_err else np.nan,
        "mean_delta": float(np.nanmean(delta[0])),
        "u_grid": u_grid,
        "phi": phi, "psi": psi, "h": h, "delta": delta,
        "q_s": result["q_s"],
    }


# -------------------------------------------------------------------
# Scenario 4: Pareto marginals
# -------------------------------------------------------------------

def run_pareto_equal_alpha(
    alpha: float = 2.5,
    sigmas: list = [1.0, 1.5],
    u_star: float = 0.5,
    n_u: int = 100,
    n_asymp: int = 12,
) -> dict:
    """Pareto with equal alpha: exact equality Phi = Psi."""

    sigmas = np.array(sigmas)
    alphas = np.array([alpha, alpha])
    q_xi_funcs, _ = build_pareto_scenario(list(alphas), list(sigmas))

    u_grid = np.linspace(u_star, 0.999, n_u)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi, psi, delta = result["phi"], result["psi"], result["delta"]

    equality_ok = float(np.nanmax(np.abs(delta))) < 1e-4

    # Asymptotic sequence — use vectorised grid computation to avoid per-u overhead
    u_asymp = 1.0 - 10.0 ** (-np.arange(1, n_asymp + 1) / 4.0)
    u_asymp = u_asymp[u_asymp < 0.9999]
    res_asymp = compute_allocations_on_grid(q_xi_funcs, u_asymp, n_points=N_POINTS)
    phi_asymp = list(res_asymp["phi"][0])
    psi_asymp = list(res_asymp["psi"][0])

    expected_limit = float(compute_pareto_asymptotic_allocation(alphas, sigmas, 0))

    return {
        "scenario": "Pareto_Equal_Alpha",
        "alpha": alpha, "sigmas": sigmas,
        "equality_ok": equality_ok,
        "max_delta": float(np.nanmax(np.abs(delta))),
        "expected_limit": expected_limit,
        "phi_asymp": phi_asymp, "psi_asymp": psi_asymp,
        "u_asymp": u_asymp,
        "u_grid": u_grid, "phi": phi, "psi": psi, "delta": delta,
        "q_s": result["q_s"],
    }


def run_pareto_unequal_alpha(
    alphas: list = [2.0, 4.0],
    sigmas: list = [1.0, 1.0],
    u_star: float = 0.5,
    n_u: int = 100,
    n_asymp: int = 12,
) -> dict:
    """Pareto with unequal alpha: dominant component has smaller alpha."""

    alphas = np.array(alphas)
    sigmas = np.array(sigmas)
    q_xi_funcs, _ = build_pareto_scenario(list(alphas), list(sigmas))

    u_grid = np.linspace(u_star, 0.999, n_u)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi, psi, delta = result["phi"], result["psi"], result["delta"]

    # alpha_min = 2 => component 0 is heaviest-tailed => should dominate
    expected_limit0 = float(compute_pareto_asymptotic_allocation(alphas, sigmas, 0))
    expected_limit1 = float(compute_pareto_asymptotic_allocation(alphas, sigmas, 1))

    # Asymptotic sequence — vectorised
    u_asymp = 1.0 - 10.0 ** (-np.arange(1, n_asymp + 1) / 4.0)
    u_asymp = u_asymp[u_asymp < 0.9999]
    res_asymp = compute_allocations_on_grid(q_xi_funcs, u_asymp, n_points=N_POINTS)
    phi0_asymp = list(res_asymp["phi"][0])
    phi1_asymp = list(res_asymp["phi"][1])

    # Concentration: phi0 -> 1, phi1 -> 0
    phi0_tail = [v for v in phi0_asymp[-5:] if not np.isnan(v)]
    concentration_ok = (len(phi0_tail) > 0 and np.mean(phi0_tail) > 0.7)

    return {
        "scenario": "Pareto_Unequal_Alpha",
        "alphas": alphas, "sigmas": sigmas,
        "expected_limit0": expected_limit0, "expected_limit1": expected_limit1,
        "concentration_ok": concentration_ok,
        "mean_phi0_tail": float(np.mean(phi0_tail)) if phi0_tail else np.nan,
        "u_asymp": u_asymp,
        "phi0_asymp": phi0_asymp, "phi1_asymp": phi1_asymp,
        "u_grid": u_grid, "phi": phi, "psi": psi, "delta": delta,
        "q_s": result["q_s"],
    }


def run_pareto_multiple_min_alpha(
    alphas: list = [2.0, 2.0, 4.0],
    sigmas: list = [1.0, 2.0, 1.0],
    u_star: float = 0.5,
    n_u: int = 100,
    n_asymp: int = 12,
) -> dict:
    """Multiple min-alpha Pareto: mass splits by sigma among minimum-alpha subset."""

    alphas = np.array(alphas)
    sigmas = np.array(sigmas)
    q_xi_funcs, _ = build_pareto_scenario(list(alphas), list(sigmas))

    u_grid = np.linspace(u_star, 0.999, n_u)
    result = compute_allocations_on_grid(q_xi_funcs, u_grid, n_points=N_POINTS)
    phi, psi, delta = result["phi"], result["psi"], result["delta"]

    # Expected limits: sigma_i/sum(sigma_j: alpha_j = alpha_min) for i in min set, else 0
    expected_limits = [float(compute_pareto_asymptotic_allocation(alphas, sigmas, i))
                       for i in range(len(alphas))]

    # Asymptotic sequence — vectorised to avoid per-u overhead
    u_asymp = 1.0 - 10.0 ** (-np.arange(1, n_asymp + 1) / 4.0)
    u_asymp = u_asymp[u_asymp < 0.9999]
    res_asymp = compute_allocations_on_grid(q_xi_funcs, u_asymp, n_points=N_POINTS)
    phi_asymp = [list(res_asymp["phi"][i]) for i in range(len(alphas))]

    # Check sigma-split
    alpha_min = alphas.min()
    min_mask = np.isclose(alphas, alpha_min)
    total_sigma_min = sigmas[min_mask].sum()
    predicted_splits = {i: (sigmas[i] / total_sigma_min if min_mask[i] else 0.0)
                        for i in range(len(alphas))}

    return {
        "scenario": "Pareto_Multiple_Min_Alpha",
        "alphas": alphas, "sigmas": sigmas,
        "expected_limits": expected_limits,
        "predicted_splits": predicted_splits,
        "u_asymp": u_asymp, "phi_asymp": phi_asymp,
        "u_grid": u_grid, "phi": phi, "psi": psi, "delta": delta,
        "q_s": result["q_s"],
    }


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_comonotonic_results(results: list, save_path: str) -> None:
    """Plot all comonotonic scenario results."""

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    fig.suptitle("Exp 3: Comonotonic Risk Experiments", fontsize=14)

    if n == 1:
        axes = axes.reshape(1, -1)

    for row, res in enumerate(results):
        u = res["u_grid"]
        phi0 = res["phi"][0]
        psi0 = res["psi"][0]
        h0 = res["h"][0] if "h" in res else None
        delta0 = res["delta"][0]

        ax0 = axes[row, 0]
        ax0.plot(u, phi0, label="Φ₁,ᵤ", color="blue", lw=1.5)
        ax0.plot(u, psi0, label="Ψ₁,ᵤ", color="red", linestyle="--", lw=1.5)
        ax0.set_title(res["scenario"][:35], fontsize=8)
        ax0.legend(fontsize=7); ax0.set_xlabel("u", fontsize=8)

        ax1 = axes[row, 1]
        if h0 is not None:
            ax1.plot(u, h0, color="green", label="h(u)=q_Xi/q_S")
            ax1.legend(fontsize=7)
        ax1.set_title("h(u) ratio", fontsize=8); ax1.set_xlabel("u", fontsize=8)

        ax2 = axes[row, 2]
        ax2.plot(u, delta0, color="purple")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_title(f"Δ = Φ - Ψ (mean={np.nanmean(delta0):.4f})", fontsize=8)
        ax2.set_xlabel("u", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_pareto_asymptotics(res_eq: dict, res_ineq: dict, save_path: str) -> None:
    """Plot Pareto asymptotic concentration."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Exp 3: Pareto Asymptotic Tail Concentration", fontsize=13)

    # Equal alpha
    ax = axes[0]
    u_a = res_eq["u_asymp"]
    ax.plot(range(len(u_a)), res_eq["phi_asymp"], label="Φ₁ (equal α)", color="blue")
    ax.plot(range(len(u_a)), res_eq["psi_asymp"], label="Ψ₁ (equal α)", color="red", linestyle="--")
    ax.axhline(res_eq["expected_limit"], color="green", lw=1, linestyle=":", label=f"ℓ₁={res_eq['expected_limit']:.3f}")
    ax.set_xlabel("Asymptotic index k"); ax.set_title("Equal alpha: Phi = Psi -> limit")
    ax.legend(fontsize=8)

    # Unequal alpha
    ax = axes[1]
    u_b = res_ineq["u_asymp"]
    ax.plot(range(len(u_b)), res_ineq["phi0_asymp"], label="Φ₁ (α=2, heavy)", color="blue")
    ax.plot(range(len(u_b)), res_ineq["phi1_asymp"], label="Φ₂ (α=4, light)", color="orange")
    ax.axhline(res_ineq["expected_limit0"], color="green", lw=1, linestyle=":", label=f"ℓ₁={res_ineq['expected_limit0']:.2f}")
    ax.axhline(res_ineq["expected_limit1"], color="red", lw=1, linestyle=":", label=f"ℓ₂={res_ineq['expected_limit1']:.2f}")
    ax.set_xlabel("Asymptotic index k"); ax.set_title("Unequal alpha: concentration on heavy tail")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_experiment_3() -> pd.DataFrame:
    """Run all Experiment 3 scenarios."""

    print("=" * 60)
    print("EXPERIMENT 3: Comonotonic Risk Experiments")
    print("=" * 60)

    print("\n[3.1] Proportional quantile equality...")
    res1 = run_proportional_quantile_equality(a=[0.3, 0.7], u_star=0.5)
    print(f"  Equality holds: {res1['equality_ok']}, max_delta={res1['max_delta']:.2e}")
    print(f"  Share errors: phi={res1['phi_share_err']:.2e}, psi={res1['psi_share_err']:.2e}")

    print("\n[3.2] Increasing h(u): Phi > Psi...")
    res2 = run_increasing_ratio(u_star=0.3)
    print(f"  h increasing: {res2['h_monotone_increasing']}, Phi1>Psi1: {res2['phi1_gt_psi1']}, mean_delta={res2['mean_delta']:.4f}")

    print("\n[3.3] Decreasing h(u): Phi < Psi...")
    res3 = run_decreasing_ratio(u_star=0.3)
    print(f"  h decreasing: {res3['h_monotone_decreasing']}, Phi1<Psi1: {res3['phi1_lt_psi1']}, mean_delta={res3['mean_delta']:.4f}")

    print("\n[3.4] Location-scale discrepancy formula...")
    res4 = run_location_scale(mus=[1.0, 2.0], sigmas=[1.0, 2.0], u_star=0.7)
    print(f"  Discrepancy mean_err={res4['discrepancy_mean_err']:.4f}, max_err={res4['discrepancy_max_err']:.4f}")

    print("\n[3.5] Pareto equal alpha (equality)...")
    res5 = run_pareto_equal_alpha(alpha=2.5, sigmas=[1.0, 1.5], u_star=0.4)
    print(f"  Equality holds: {res5['equality_ok']}, max_delta={res5['max_delta']:.2e}")

    print("\n[3.6] Pareto unequal alpha (dominance & concentration)...")
    res6 = run_pareto_unequal_alpha(alphas=[2.0, 4.0], sigmas=[1.0, 1.0], u_star=0.4)
    print(f"  Concentration ok: {res6['concentration_ok']}, mean_phi0_tail={res6['mean_phi0_tail']:.4f}")

    print("\n[3.7] Pareto multiple min-alpha (sigma-split)...")
    res7 = run_pareto_multiple_min_alpha(alphas=[2.0, 2.0, 4.0], sigmas=[1.0, 2.0, 1.0])
    print(f"  Expected limits: {[f'{v:.3f}' for v in res7['expected_limits']]}")
    print(f"  Predicted splits: {res7['predicted_splits']}")

    plot_comonotonic_results(
        [res1, res2, res3, res4],
        os.path.join(RESULTS_DIR, "exp3_comonotonic_main.png")
    )
    plot_pareto_asymptotics(res5, res6, os.path.join(RESULTS_DIR, "exp3_pareto_asymptotics.png"))

    summary = pd.DataFrame([
        {"Scenario": "Proportional Quantile Equality", "Equality_ok": res1["equality_ok"],
         "Max_delta": res1["max_delta"], "Note": f"shares={res1['expected_shares']}"},
        {"Scenario": "Increasing h => Phi>Psi", "Equality_ok": False,
         "Max_delta": abs(res2["mean_delta"]), "Note": f"h_incr={res2['h_monotone_increasing']}"},
        {"Scenario": "Decreasing h => Phi<Psi", "Equality_ok": False,
         "Max_delta": abs(res3["mean_delta"]), "Note": f"h_decr={res3['h_monotone_decreasing']}"},
        {"Scenario": "Location-Scale Discrepancy", "Equality_ok": abs(res4["mean_delta"]) < 1e-3,
         "Max_delta": res4["discrepancy_max_err"], "Note": "closed-form match"},
        {"Scenario": "Pareto Equal Alpha", "Equality_ok": res5["equality_ok"],
         "Max_delta": res5["max_delta"], "Note": f"alpha={res5['alpha']}"},
        {"Scenario": "Pareto Unequal Alpha", "Equality_ok": False,
         "Max_delta": float(np.nanmax(np.abs(res6["delta"]))),
         "Note": f"conc_ok={res6['concentration_ok']}"},
        {"Scenario": "Pareto Multiple Min-Alpha", "Equality_ok": False,
         "Max_delta": float(np.nanmax(np.abs(res7["delta"]))),
         "Note": f"limits={[f'{v:.2f}' for v in res7['expected_limits']]}"},
    ])

    summary.to_csv(os.path.join(RESULTS_DIR, "exp3_summary.csv"), index=False)
    print("\nSummary saved to results/exp3_summary.csv")
    return summary


if __name__ == "__main__":
    summary = run_experiment_3()
    print("\n" + summary.to_string(index=False))
