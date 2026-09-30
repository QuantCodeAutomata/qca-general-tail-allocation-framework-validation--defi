"""
Comonotonic risk allocation formulas.

Framework: X_i = q_{X_i}(U) for a common U ~ Uniform(0,1).
Key identity: q_S(u) = sum_i q_{X_i}(u) (comonotonic aggregate quantile).

Allocation measures implemented:
  Phi_{i,u}  — TCE-ratio allocation (integral of quantile / integral of aggregate)
  Psi_{i,u}  — Quantile-ratio allocation (integral of pointwise ratios)
  h_i(u)     — Pointwise ratio q_{X_i}(u) / q_S(u)
  Delta_{i,u} = Phi_{i,u} - Psi_{i,u}  (discrepancy)
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate
from typing import Callable, List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _eval_q(q_func: Callable, v: np.ndarray) -> np.ndarray:
    """
    Evaluate quantile function q_func on array v, using vectorised call when
    possible and falling back to element-wise iteration otherwise.

    Parameters
    ----------
    q_func : Callable
        Quantile function; may accept either a scalar or a numpy array.
    v : np.ndarray
        Probability levels at which to evaluate q_func.

    Returns
    -------
    np.ndarray
        q_func evaluated at every element of v.
    """
    try:
        result = q_func(v)
        if np.ndim(result) == 0:
            raise TypeError
        return np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        return np.array([q_func(vi) for vi in v], dtype=float)


# ---------------------------------------------------------------------------
# Aggregate quantile
# ---------------------------------------------------------------------------

def compute_comonotonic_aggregate_quantile(
    q_xi_funcs: List[Callable], u: float
) -> float:
    """
    Compute comonotonic aggregate quantile q_S(u) = sum_i q_{X_i}(u).

    In the comonotonic framework X_i = q_{X_i}(U) for a shared U ~ Uniform(0,1),
    so S = sum_i X_i has quantile q_S(u) = sum_i q_{X_i}(u).

    Parameters
    ----------
    q_xi_funcs : List[Callable]
        Quantile functions q_{X_i}: [0,1) -> R, one per component.
    u : float
        Probability level in [0, 1).

    Returns
    -------
    float
        Aggregate comonotonic quantile q_S(u).
    """
    return sum(q(u) for q in q_xi_funcs)


def compute_comonotonic_aggregate_quantile_grid(
    q_xi_funcs: List[Callable], u_grid: np.ndarray
) -> np.ndarray:
    """
    Compute comonotonic aggregate quantile q_S(u) for every u in a grid (vectorized).

    Parameters
    ----------
    q_xi_funcs : List[Callable]
        Quantile functions q_{X_i}: [0,1) -> R, one per component.
    u_grid : np.ndarray
        Array of probability levels in [0, 1).

    Returns
    -------
    np.ndarray
        q_S(u) for each u in u_grid; same length as u_grid.
    """
    result = np.zeros(len(u_grid))
    for q in q_xi_funcs:
        result += _eval_q(q, u_grid)
    return result


# ---------------------------------------------------------------------------
# Allocation measures: Phi and Psi
# ---------------------------------------------------------------------------

def compute_phi_comonotonic(
    q_xi_func: Callable,
    q_xi_funcs_all: List[Callable],
    u: float,
    n_points: int = 5000,
    u_max: float = 1.0 - 1e-8,
) -> float:
    """
    Compute Phi_{i,u} = int_u^1 q_{X_i}(v) dv / int_u^1 q_S(v) dv.

    This is the TCE-ratio (Tail Conditional Expectation) allocation. The common
    factor (1-u) in numerator and denominator cancels, leaving the integral ratio.
    Integration uses dense trapezoidal quadrature from u to u_max.

    Parameters
    ----------
    q_xi_func : Callable
        Quantile function of component i.
    q_xi_funcs_all : List[Callable]
        All component quantile functions (summed to form q_S).
    u : float
        Lower integration limit / confidence level.
    n_points : int
        Number of trapezoidal quadrature nodes from u to u_max.
    u_max : float
        Upper integration limit, slightly below 1 to avoid singularities.

    Returns
    -------
    float
        Phi_{i,u}; np.nan when u >= u_max or the denominator integral is zero.
    """
    if u >= u_max:
        return np.nan

    v = np.linspace(u, u_max, n_points)

    qi_vals = _eval_q(q_xi_func, v)
    qs_vals = np.zeros(n_points)
    for q in q_xi_funcs_all:
        qs_vals += _eval_q(q, v)

    num = np.trapz(qi_vals, v)
    den = np.trapz(qs_vals, v)

    if den == 0.0:
        warnings.warn(
            f"Denominator int_u^1 q_S(v) dv = 0 at u={u}; returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    return num / den


def compute_psi_comonotonic(
    q_xi_func: Callable,
    q_xi_funcs_all: List[Callable],
    u: float,
    n_points: int = 5000,
    u_max: float = 1.0 - 1e-8,
) -> float:
    """
    Compute Psi_{i,u} = (1/(1-u)) * int_u^1 q_{X_i}(v) / q_S(v) dv.

    This is the quantile-ratio allocation: the conditional expectation of the
    pointwise share q_{X_i}(U)/q_S(U) given U > u. Where q_S(v) = 0 the
    integrand is treated as 0 and a RuntimeWarning is raised.

    Parameters
    ----------
    q_xi_func : Callable
        Quantile function of component i.
    q_xi_funcs_all : List[Callable]
        All component quantile functions (summed to form q_S).
    u : float
        Confidence level.
    n_points : int
        Number of trapezoidal quadrature nodes from u to u_max.
    u_max : float
        Upper integration limit, slightly below 1 to avoid singularities.

    Returns
    -------
    float
        Psi_{i,u}; np.nan when u >= u_max.
    """
    if u >= u_max:
        return np.nan

    v = np.linspace(u, u_max, n_points)

    qi_vals = _eval_q(q_xi_func, v)
    qs_vals = np.zeros(n_points)
    for q in q_xi_funcs_all:
        qs_vals += _eval_q(q, v)

    zero_mask = qs_vals == 0.0
    if np.any(zero_mask):
        warnings.warn(
            "q_S(v) = 0 at some integration points; integrand set to 0 there.",
            RuntimeWarning,
            stacklevel=2,
        )

    ratio = np.where(~zero_mask, qi_vals / qs_vals, 0.0)
    return np.trapz(ratio, v) / (1.0 - u)


# ---------------------------------------------------------------------------
# Pointwise ratio h(u)
# ---------------------------------------------------------------------------

def compute_h_ratio(
    q_xi_func: Callable,
    q_xi_funcs_all: List[Callable],
    u_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the pointwise quantile ratio h(u) = q_{X_i}(u) / q_S(u) on a grid.

    When h_i(u) is constant in u for all i, the components have proportional
    quantile functions and Phi_i = Psi_i holds exactly.

    Parameters
    ----------
    q_xi_func : Callable
        Quantile function of component i.
    q_xi_funcs_all : List[Callable]
        All component quantile functions (summed to form q_S).
    u_grid : np.ndarray
        Grid of probability levels in [0, 1).

    Returns
    -------
    np.ndarray
        h(u) for each u in u_grid; np.nan wherever q_S(u) = 0.
    """
    qi_vals = _eval_q(q_xi_func, u_grid)
    qs_vals = np.zeros(len(u_grid))
    for q in q_xi_funcs_all:
        qs_vals += _eval_q(q, u_grid)

    return np.where(qs_vals != 0.0, qi_vals / qs_vals, np.nan)


# ---------------------------------------------------------------------------
# Tail risk measures: ES and HS
# ---------------------------------------------------------------------------

def compute_es_comonotonic(
    q_xi_funcs_all: List[Callable],
    u: float,
    n_points: int = 5000,
    u_max: float = 1.0 - 1e-8,
) -> float:
    """
    Compute Expected Shortfall ES_u(S) = (1/(1-u)) * int_u^1 q_S(v) dv.

    Parameters
    ----------
    q_xi_funcs_all : List[Callable]
        All component quantile functions (summed to form q_S).
    u : float
        Confidence level.
    n_points : int
        Number of trapezoidal quadrature nodes.
    u_max : float
        Upper integration limit, slightly below 1.

    Returns
    -------
    float
        Expected Shortfall of the aggregate S at level u; np.nan if u >= u_max.
    """
    if u >= u_max:
        return np.nan

    v = np.linspace(u, u_max, n_points)
    qs_vals = np.zeros(n_points)
    for q in q_xi_funcs_all:
        qs_vals += _eval_q(q, v)

    return np.trapz(qs_vals, v) / (1.0 - u)


def compute_hs_u(
    q_xi_funcs_all: List[Callable],
    u: float,
    n_points: int = 5000,
    u_max: float = 1.0 - 1e-8,
) -> float:
    """
    Compute tail harmonic mean HS_u(S) = [(1/(1-u)) int_u^1 q_S(v)^{-1} dv]^{-1}.

    HS_u(S) is the harmonic mean of q_S over the tail (u, 1). It is only
    well-defined when q_S(v) > 0 for all v in [u, 1); np.nan is returned
    and a warning is raised if this condition is violated.

    Parameters
    ----------
    q_xi_funcs_all : List[Callable]
        All component quantile functions (summed to form q_S).
    u : float
        Confidence level.
    n_points : int
        Number of trapezoidal quadrature nodes.
    u_max : float
        Upper integration limit, slightly below 1.

    Returns
    -------
    float
        Tail harmonic mean HS_u(S); np.nan if u >= u_max or q_S(v) <= 0 anywhere.
    """
    if u >= u_max:
        return np.nan

    v = np.linspace(u, u_max, n_points)
    qs_vals = np.zeros(n_points)
    for q in q_xi_funcs_all:
        qs_vals += _eval_q(q, v)

    if np.any(qs_vals <= 0.0):
        warnings.warn(
            "q_S(v) <= 0 in [u, u_max]; HS_u requires strictly positive q_S.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    harmonic_mean_inv = np.trapz(1.0 / qs_vals, v) / (1.0 - u)

    if harmonic_mean_inv == 0.0:
        warnings.warn("Harmonic-mean denominator is zero.", RuntimeWarning, stacklevel=2)
        return np.nan

    return 1.0 / harmonic_mean_inv


def compute_es_hs_on_grid(
    q_xi_funcs_all: List[Callable],
    u_grid: np.ndarray,
    n_points: int = 5000,
    u_max: float = 1.0 - 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ES_u(S) and HS_u(S) for every u in u_grid in a single pass.

    This batch version avoids the O(n_u * n_points) re-evaluation of q_S that
    occurs when calling ``compute_es_comonotonic`` and ``compute_hs_u``
    individually for each u.  Instead q_S is evaluated once on a dense grid
    that spans the entire integration range, and suffix integrals are extracted
    per u via searchsorted.

    Parameters
    ----------
    q_xi_funcs_all : List[Callable]
        All component quantile functions that sum to q_S.
    u_grid : np.ndarray
        Sorted array of confidence levels in [0, 1).
    n_points : int
        Dense-grid resolution; default 5 000.
    u_max : float
        Upper integration limit, slightly below 1.

    Returns
    -------
    es_arr : np.ndarray
        ES_u(S) at each u in u_grid; np.nan for u >= u_max.
    hs_arr : np.ndarray
        HS_u(S) at each u in u_grid; np.nan wherever q_S is non-positive on the
        sub-interval or u >= u_max.
    """
    u_grid = np.asarray(u_grid, dtype=float)
    n_u = len(u_grid)
    es_arr = np.full(n_u, np.nan)
    hs_arr = np.full(n_u, np.nan)

    # Build dense grid + insert every u as an exact node.
    v_dense = np.linspace(float(u_grid[0]), u_max, n_points)
    valid_u = u_grid[u_grid < u_max]
    v_combined = np.unique(np.concatenate([v_dense, valid_u]))
    n_v = len(v_combined)

    # Evaluate q_S once on the combined grid.
    qs_vals = np.zeros(n_v)
    for q in q_xi_funcs_all:
        qs_vals += _eval_q(q, v_combined)

    dv = np.diff(v_combined)  # (n_v-1,)

    # Suffix integral of q_S: suffix_qs[j] = int_{v[j]}^{u_max} q_S(v) dv
    incr_qs = 0.5 * (qs_vals[:-1] + qs_vals[1:]) * dv
    cum_qs = np.concatenate([[0.0], np.cumsum(incr_qs)])  # (n_v,)
    total_qs = cum_qs[-1]
    suffix_qs = total_qs - cum_qs  # (n_v,)

    # Suffix integral of 1/q_S where q_S > 0; 0 elsewhere.
    inv_qs = np.where(qs_vals > 0.0, 1.0 / qs_vals, 0.0)
    incr_inv = 0.5 * (inv_qs[:-1] + inv_qs[1:]) * dv
    cum_inv = np.concatenate([[0.0], np.cumsum(incr_inv)])  # (n_v,)
    total_inv = cum_inv[-1]
    suffix_inv = total_inv - cum_inv  # (n_v,)

    # Flag nodes where q_S <= 0 so HS can be marked invalid.
    nonpos_mask = qs_vals <= 0.0
    # Cumulative count of non-positive nodes for range checks.
    cum_nonpos = np.concatenate([[0], np.cumsum(nonpos_mask)])  # (n_v+1,)

    # Map each u in u_grid to its index in v_combined.
    valid_mask = u_grid < u_max
    valid_j = np.where(valid_mask)[0]
    u_valid = u_grid[valid_mask]
    u_idxs = np.searchsorted(v_combined, u_valid)
    u_idxs = np.minimum(u_idxs, n_v - 1)

    width = 1.0 - u_valid  # (n_valid,)

    # ES
    es_arr[valid_j] = suffix_qs[u_idxs] / width

    # HS: valid only when q_S > 0 throughout [u, u_max]
    nonpos_in_range = cum_nonpos[n_v] - cum_nonpos[u_idxs]
    hs_valid = (nonpos_in_range == 0) & (width > 0)

    inv_suffix = suffix_inv[u_idxs]
    mean_inv = np.where(hs_valid, inv_suffix / width, 0.0)
    hs_arr[valid_j] = np.where(hs_valid & (mean_inv > 0.0), 1.0 / mean_inv, np.nan)

    return es_arr, hs_arr


# ---------------------------------------------------------------------------
# Location-scale discrepancy
# ---------------------------------------------------------------------------

def compute_location_scale_discrepancy(
    mu_i: float,
    sigma_i: float,
    mu_s: float,
    sigma_s: float,
    es_u: float,
    hs_u: float,
) -> float:
    """
    Closed-form discrepancy Phi_{i,u} - Psi_{i,u} for location-scale families.

    When X_i = mu_i + sigma_i * Z for a common Z, the analytical formula is:

        Phi_{i,u} - Psi_{i,u}
            = ((sigma_i * mu_S - sigma_S * mu_i) / sigma_S) * (1/HS_u(S) - 1/ES_u(S))

    Parameters
    ----------
    mu_i : float
        Location parameter of component i.
    sigma_i : float
        Scale parameter of component i.
    mu_s : float
        Location parameter of aggregate S (= sum_j mu_j).
    sigma_s : float
        Scale parameter of aggregate S (= sum_j sigma_j).
    es_u : float
        Expected Shortfall of S at level u.
    hs_u : float
        Tail harmonic mean of S at level u.

    Returns
    -------
    float
        Phi_{i,u} - Psi_{i,u} discrepancy.
    """
    coeff = (sigma_i * mu_s - sigma_s * mu_i) / sigma_s
    return coeff * (1.0 / hs_u - 1.0 / es_u)


# ---------------------------------------------------------------------------
# Pareto distribution utilities
# ---------------------------------------------------------------------------

def compute_pareto_quantile(
    alpha: float, sigma: float, u: "float | np.ndarray"
) -> "float | np.ndarray":
    """
    Compute Pareto quantile q_{X_i}(u) = sigma * (1 - u)^{-1/alpha}.

    Standard Pareto with tail index alpha > 0 and scale sigma > 0. The quantile
    diverges to +infinity as u -> 1 at rate (1-u)^{-1/alpha}; heavier tails
    (smaller alpha) produce faster divergence.

    Parameters
    ----------
    alpha : float
        Tail index (must satisfy alpha > 0).
    sigma : float
        Scale parameter (must satisfy sigma > 0).
    u : float or np.ndarray
        Probability level(s) in [0, 1).

    Returns
    -------
    float or np.ndarray
        Pareto quantile at level(s) u; same shape as input.
    """
    u_arr = np.asarray(u, dtype=float)
    result = sigma * (1.0 - u_arr) ** (-1.0 / alpha)
    return float(result) if u_arr.ndim == 0 else result


def compute_pareto_asymptotic_allocation(
    alphas: np.ndarray,
    sigmas: np.ndarray,
    i: int,
) -> float:
    """
    Asymptotic limit of Phi_{i,u} and Psi_{i,u} as u -> 1 for Pareto components.

    As u -> 1 both allocations converge to the same limit determined solely by the
    heaviest-tailed components (smallest alpha):

        ell_i = sigma_i * 1_{alpha_i = alpha_min} / sum_{j: alpha_j = alpha_min} sigma_j

    Components whose tail index exceeds alpha_min contribute 0 in the limit because
    their quantiles are eventually negligible relative to the dominant components.

    Parameters
    ----------
    alphas : np.ndarray
        Tail indices for each component (alpha_j > 0).
    sigmas : np.ndarray
        Scale parameters for each component (sigma_j > 0).
    i : int
        0-based index of the component whose limit is requested.

    Returns
    -------
    float
        Asymptotic allocation ell_i; 0.0 if alpha_i > alpha_min.
    """
    alphas = np.asarray(alphas, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    alpha_min = np.min(alphas)

    if not np.isclose(alphas[i], alpha_min):
        return 0.0

    dominant_mask = np.isclose(alphas, alpha_min)
    sum_dominant = np.sum(sigmas[dominant_mask])

    if sum_dominant == 0.0:
        warnings.warn(
            "Sum of dominant-component sigmas is zero.", RuntimeWarning, stacklevel=2
        )
        return np.nan

    return float(sigmas[i] / sum_dominant)


# ---------------------------------------------------------------------------
# Grid-based allocation computation
# ---------------------------------------------------------------------------

def compute_allocations_on_grid(
    q_xi_funcs: List[Callable],
    u_grid: np.ndarray,
    n_points: int = 3000,
) -> Dict:
    """
    Compute Phi, Psi, h, and Delta for all components at every u on a grid.

    Quantile functions are evaluated once on a single combined dense grid
    (spanning [u_grid[0], 1-1e-8]) to avoid redundant calls. For each u in
    u_grid the integration from u to u_max reuses the pre-evaluated values.

    Parameters
    ----------
    q_xi_funcs : List[Callable]
        Quantile functions for each component.
    u_grid : np.ndarray
        Grid of confidence levels in [0, 1).
    n_points : int
        Number of nodes in the dense integration grid.

    Returns
    -------
    Dict with keys:
        'phi'   : np.ndarray, shape (n_comp, n_u) — TCE-ratio allocations
        'psi'   : np.ndarray, shape (n_comp, n_u) — quantile-ratio allocations
        'h'     : np.ndarray, shape (n_comp, n_u) — pointwise quantile ratios
        'delta' : np.ndarray, shape (n_comp, n_u) — phi - psi discrepancies
        'q_s'   : np.ndarray, shape (n_u,)         — aggregate quantile on grid
    """
    n_comp = len(q_xi_funcs)
    n_u = len(u_grid)
    u_max = 1.0 - 1e-8

    # Dense grid covering the full integration range; include exact u_grid points.
    v_dense = np.linspace(float(u_grid[0]), u_max, n_points)
    valid_u = u_grid[u_grid < u_max]
    v_combined = np.unique(np.concatenate([v_dense, valid_u]))
    n_v = len(v_combined)

    # Pre-evaluate all quantile functions on the combined grid: shape (n_comp, n_v)
    q_vals = np.array([_eval_q(q, v_combined) for q in q_xi_funcs])
    qs_vals = q_vals.sum(axis=0)  # shape (n_v,)

    phi = np.full((n_comp, n_u), np.nan)
    psi = np.full((n_comp, n_u), np.nan)
    h = np.full((n_comp, n_u), np.nan)

    # ------------------------------------------------------------------
    # Suffix (tail) integrals — computed once for the entire v_combined
    # grid, then extracted at each u in u_grid without per-u trapz calls.
    #
    # For a function f evaluated at v_combined[0..n_v-1]:
    #   suffix_F[j] = int_{v_combined[j]}^{u_max} f(v) dv
    #               ≈ sum_{l=j}^{n_v-2} 0.5*(f[l]+f[l+1])*(v[l+1]-v[l])
    #
    # This equals total_F - cumulative_F[j] where
    #   cumulative_F[j] = int_{v_combined[0]}^{v_combined[j]} f(v) dv.
    # We use np.cumsum on the trapezoidal weights for each f.
    # ------------------------------------------------------------------
    dv = np.diff(v_combined)                        # shape (n_v-1,)

    # Trapezoidal increments for q_S: shape (n_v-1,)
    incr_qs = 0.5 * (qs_vals[:-1] + qs_vals[1:]) * dv
    cum_qs = np.concatenate([[0.0], np.cumsum(incr_qs)])  # shape (n_v,)
    total_qs = cum_qs[-1]
    suffix_qs = total_qs - cum_qs                   # shape (n_v,)

    # For each component k: trapezoidal increments and suffix integrals
    # shape (n_comp, n_v-1) and (n_comp, n_v)
    incr_qi = 0.5 * (q_vals[:, :-1] + q_vals[:, 1:]) * dv  # (n_comp, n_v-1)
    cum_qi = np.concatenate(
        [np.zeros((n_comp, 1)), np.cumsum(incr_qi, axis=1)], axis=1
    )  # (n_comp, n_v)
    total_qi = cum_qi[:, -1:]                       # (n_comp, 1)
    suffix_qi = total_qi - cum_qi                   # (n_comp, n_v)

    # Psi numerator: ratio = q_i / q_S pointwise; suffix integral of ratio
    # shape (n_v,) for each k, but computed per-component below
    ratio_all = np.where(
        qs_vals[np.newaxis, :] != 0.0,
        q_vals / qs_vals[np.newaxis, :],
        0.0,
    )  # (n_comp, n_v)
    incr_ratio = 0.5 * (ratio_all[:, :-1] + ratio_all[:, 1:]) * dv  # (n_comp, n_v-1)
    cum_ratio = np.concatenate(
        [np.zeros((n_comp, 1)), np.cumsum(incr_ratio, axis=1)], axis=1
    )  # (n_comp, n_v)
    total_ratio = cum_ratio[:, -1:]
    suffix_ratio = total_ratio - cum_ratio          # (n_comp, n_v)

    # Batch-index all u in u_grid at once — no Python loop needed.
    valid_mask = u_grid < u_max                         # (n_u,)
    valid_j = np.where(valid_mask)[0]                   # indices into u_grid
    u_valid = u_grid[valid_mask]                        # (n_valid,)

    # Each u was inserted as an exact node; searchsorted finds it exactly.
    u_idxs = np.searchsorted(v_combined, u_valid)      # (n_valid,)
    clamp = u_idxs < n_v
    u_idxs = np.where(clamp, u_idxs, 0)                # avoid out-of-bounds

    den_valid = suffix_qs[u_idxs]                      # (n_valid,)
    den_ok = (den_valid != 0.0) & clamp

    # Phi: (n_comp, n_valid)
    phi[:, valid_j] = np.where(
        den_ok[np.newaxis, :],
        suffix_qi[:, u_idxs] / np.where(den_ok, den_valid, 1.0)[np.newaxis, :],
        np.nan,
    )

    # Psi: (n_comp, n_valid)
    width = 1.0 - u_valid                              # (n_valid,)
    psi[:, valid_j] = np.where(
        (width > 0) & clamp,
        suffix_ratio[:, u_idxs] / np.where(width > 0, width, 1.0)[np.newaxis, :],
        np.nan,
    )

    # h(u): pointwise ratio at each u node
    qs_u_valid = qs_vals[u_idxs]                       # (n_valid,)
    qs_ok = (qs_u_valid != 0.0) & clamp
    h[:, valid_j] = np.where(
        qs_ok[np.newaxis, :],
        q_vals[:, u_idxs] / np.where(qs_ok, qs_u_valid, 1.0)[np.newaxis, :],
        np.nan,
    )

    q_s = compute_comonotonic_aggregate_quantile_grid(q_xi_funcs, u_grid)

    return {
        "phi": phi,
        "psi": psi,
        "h": h,
        "delta": phi - psi,
        "q_s": q_s,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_proportional_quantile_equality(
    q_xi_funcs: List[Callable],
    u_grid: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """
    Check whether all h_i(u) = q_{X_i}(u)/q_S(u) are constant over the grid.

    Constant h_i(u) is the necessary and sufficient condition for proportional
    quantile functions, which implies Phi_i = Psi_i for all i and u.

    Parameters
    ----------
    q_xi_funcs : List[Callable]
        Quantile functions for each component.
    u_grid : np.ndarray
        Grid of probability levels in [0, 1).
    tol : float
        Maximum allowed range (max - min) across the grid; default 1e-6.

    Returns
    -------
    bool
        True if every component ratio h_i(u) is approximately constant over u_grid.
    """
    for q_xi_func in q_xi_funcs:
        h = compute_h_ratio(q_xi_func, q_xi_funcs, u_grid)
        valid = ~np.isnan(h)
        if not np.any(valid):
            continue
        if np.ptp(h[valid]) > tol:
            return False
    return True


def check_ratio_monotonicity(
    h_grid: np.ndarray,
    u_grid: np.ndarray,
) -> Dict:
    """
    Characterise the monotonicity of h(u) = q_{X_i}(u)/q_S(u) on the grid.

    Uses first differences of the non-nan values with an absolute tolerance of
    1e-10 to handle floating-point rounding in near-constant sequences.

    Parameters
    ----------
    h_grid : np.ndarray
        Values of h(u) at each point of u_grid.
    u_grid : np.ndarray
        Grid of probability levels corresponding to h_grid (unused directly,
        retained for interface consistency and future weighted-difference use).

    Returns
    -------
    Dict with keys:
        'monotone_increasing' : bool — h is non-decreasing (diffs >= -tol)
        'monotone_decreasing' : bool — h is non-increasing (diffs <= +tol)
        'is_constant'         : bool — h is constant (|diffs| <= tol)
    """
    _tol = 1e-10

    valid = ~np.isnan(h_grid)
    h_valid = h_grid[valid]

    if len(h_valid) < 2:
        return {
            "monotone_increasing": True,
            "monotone_decreasing": True,
            "is_constant": True,
        }

    diffs = np.diff(h_valid)

    return {
        "monotone_increasing": bool(np.all(diffs >= -_tol)),
        "monotone_decreasing": bool(np.all(diffs <= _tol)),
        "is_constant": bool(np.all(np.abs(diffs) <= _tol)),
    }


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def build_pareto_scenario(
    alphas: List[float],
    sigmas: List[float],
) -> Tuple[List[Callable], List[Callable]]:
    """
    Build Pareto quantile functions q_{X_i}(u) for given tail indices and scales.

    Each component quantile is q_{X_i}(u) = sigma_i * (1 - u)^{-1/alpha_i}.
    A closure is used for each component so that alpha_i and sigma_i are
    correctly captured.

    Parameters
    ----------
    alphas : List[float]
        Tail indices alpha_i > 0 for each component.
    sigmas : List[float]
        Scale parameters sigma_i > 0 for each component.

    Returns
    -------
    Tuple[List[Callable], List[Callable]]
        (q_xi_funcs, ppf_list): both lists contain identical Pareto quantile
        callables. q_xi_funcs is the primary list for allocation routines;
        ppf_list provides a separate reference for direct PPF queries.
    """
    def _make_pareto_q(alpha: float, sigma: float) -> Callable:
        def q(u: float) -> float:
            return compute_pareto_quantile(alpha, sigma, u)
        return q

    q_xi_funcs = [_make_pareto_q(a, s) for a, s in zip(alphas, sigmas)]
    ppf_list = [_make_pareto_q(a, s) for a, s in zip(alphas, sigmas)]

    return q_xi_funcs, ppf_list


# ---------------------------------------------------------------------------
# Location-scale scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocationScaleScenario:
    """
    Location-scale family scenario for comonotonic risk allocation.

    Models each component as X_i = mu_i + sigma_i * Z for a common
    standardised baseline Z with quantile function z_func. Under comonotonicity,
    q_S(u) = mu_S + sigma_S * z_func(u) where mu_S = sum_i mu_i and
    sigma_S = sum_i sigma_i.

    Attributes
    ----------
    mus : np.ndarray
        Location parameters for each component.
    sigmas : np.ndarray
        Scale parameters for each component (must be positive).
    z_func : Callable
        Quantile function of the common standardised baseline Z: [0,1) -> R.
    """

    mus: np.ndarray
    sigmas: np.ndarray
    z_func: Callable  # common baseline quantile function z(u)

    def __post_init__(self) -> None:
        self.mus = np.asarray(self.mus, dtype=float)
        self.sigmas = np.asarray(self.sigmas, dtype=float)

    def q_xi(self, i: int, u: float) -> float:
        """Quantile of component i: q_{X_i}(u) = mu_i + sigma_i * z(u)."""
        return float(self.mus[i]) + float(self.sigmas[i]) * self.z_func(u)

    def q_s(self, u: float) -> float:
        """Quantile of aggregate S: q_S(u) = mu_S + sigma_S * z(u)."""
        return self.mu_s + self.sigma_s * self.z_func(u)

    @property
    def mu_s(self) -> float:
        """Location of aggregate S: mu_S = sum_i mu_i."""
        return float(np.sum(self.mus))

    @property
    def sigma_s(self) -> float:
        """Scale of aggregate S: sigma_S = sum_i sigma_i."""
        return float(np.sum(self.sigmas))
