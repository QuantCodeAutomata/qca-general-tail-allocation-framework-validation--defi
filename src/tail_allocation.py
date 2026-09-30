"""
tail_allocation.py
==================
Core tail-allocation framework module.

Implements the nonparametric tail-allocation machinery:
  - Left-continuous quantile (VaR):   q(u) = inf{x : F_S(x) >= u}
  - Expected Shortfall:               ES_u(S) = (1/(1-u)) integral_u^1 q(v) dv
  - Tail-to-quantile multiplier:      W_u = ES_u(S) / q(u)
  - Randomization parameter:          theta_u = (F_S(q(u)) - u) / P(S = q(u))
  - Tail indicator:                   pi_u(s) = 1_{s > q(u)} + theta_u 1_{s = q(u)}
  - Proportional ES allocation:       Phi_{i,u} = E[X_i pi_u(S)] / E[S pi_u(S)]
  - Proportional GTE allocation:      Psi_{i,u} = E[R_i pi_u(S)] / E[pi_u(S)]
  - Conditional quantile ratio:       h(u) = q_{X_i}(u) / q_S(u)

Covariance identity:
  Phi_{i,u} - Psi_{i,u} = Cov_u(q_S(V), h(V)) / E_u[q_S(V)]
  where V ~ Uniform(u, 1).
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize
from typing import Callable, Optional, Tuple, List, Dict
import warnings
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Left-continuous quantile (VaR)
# ---------------------------------------------------------------------------

def compute_left_quantile(
    cdf_func: Callable[[float], float],
    u: float,
    x_min: float,
    x_max: float,
    tol: float = 1e-10,
) -> float:
    """Compute the left-continuous quantile q(u) = inf{x : F_S(x) >= u}.

    This is the Value-at-Risk at level u. The function inverts the CDF using
    ``scipy.optimize.brentq`` for continuous distributions.  For discrete CDFs
    the result is a real number in the interior of the atom interval; use
    ``compute_theta_u`` to obtain the randomization weight theta_u.

    Parameters
    ----------
    cdf_func:
        Right-continuous CDF F_S evaluated at a scalar x.
    u:
        Probability level in [0, 1].
    x_min:
        Lower bound of the support (returned for u = 0).
    x_max:
        Upper bound of the support (returned for u = 1).
    tol:
        Absolute tolerance passed to ``brentq``.

    Returns
    -------
    float
        q(u), the left-continuous quantile at level u.
    """
    if u <= 0.0:
        return float(x_min)
    if u >= 1.0:
        return float(x_max)

    # If the left endpoint already satisfies F(x_min) >= u, return x_min.
    f_min = cdf_func(x_min)
    if f_min >= u:
        return float(x_min)

    # Guard against x_max being outside the effective support.
    f_max = cdf_func(x_max)
    if f_max < u:
        warnings.warn(
            f"CDF(x_max={x_max}) = {f_max:.6g} < u = {u:.6g}; "
            "returning x_max as best approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(x_max)

    # Brentq: find root of F(x) - u = 0 on (x_min, x_max).
    # For continuous F this yields q(u) exactly.
    result = optimize.brentq(
        lambda x: cdf_func(x) - u,
        x_min,
        x_max,
        xtol=tol,
        full_output=False,
    )
    return float(result)


# ---------------------------------------------------------------------------
# 2. Vectorised quantile grid
# ---------------------------------------------------------------------------

def compute_quantile_grid(
    cdf_func: Callable[[float], float],
    u_grid: np.ndarray,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Compute q(u) for every u in ``u_grid``.

    Parameters
    ----------
    cdf_func:
        Right-continuous CDF F_S(x).
    u_grid:
        1-D array of probability levels in [0, 1].
    x_min:
        Lower support bound.
    x_max:
        Upper support bound.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(u_grid),)`` with q(u) for each u.
    """
    return np.array(
        [compute_left_quantile(cdf_func, u, x_min, x_max) for u in u_grid],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# 3. Expected Shortfall
# ---------------------------------------------------------------------------

def compute_es(q_grid: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Compute ES_u(S) = (1/(1-u)) integral_u^1 q(v) dv for all u in u_grid.

    The integral is evaluated via the trapezoidal rule on the supplied grid.
    A tail correction q_grid[-1] * (1 - u_grid[-1]) is added to account for
    the region [u_grid[-1], 1] that may lie outside the grid.  When u is
    extremely close to 1 (denominator < 1e-15) the function returns q(u)
    directly, which is the correct limit.

    Parameters
    ----------
    q_grid:
        Array of quantile values q(u_grid[i]) for i = 0, …, n-1.
    u_grid:
        Corresponding probability levels, strictly increasing in [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape ``(len(u_grid),)`` with ES_u(S).
    """
    n = len(u_grid)

    # Trapezoid pieces: integral between consecutive grid points.
    du = np.diff(u_grid)                             # shape (n-1,)
    trap = 0.5 * (q_grid[:-1] + q_grid[1:]) * du    # shape (n-1,)

    # right_cumsum[i] = integral from u_grid[i] to u_grid[-1] of q(v) dv.
    # Computed by reversing the cumulative sum of trap pieces.
    right_cumsum = np.zeros(n, dtype=float)
    right_cumsum[:-1] = np.cumsum(trap[::-1])[::-1]

    # Tail correction: approximate q as constant q_grid[-1] beyond the grid.
    tail_integral = float(q_grid[-1]) * (1.0 - float(u_grid[-1]))

    # Total integral from u_grid[i] to 1.
    total_from_u = right_cumsum + tail_integral

    es = np.empty(n, dtype=float)
    denom = 1.0 - u_grid
    safe = denom > 1e-15
    es[safe] = total_from_u[safe] / denom[safe]
    # Limit as u → 1: ES_u → q(u).
    es[~safe] = q_grid[~safe]

    return es


# ---------------------------------------------------------------------------
# 4. Tail-to-quantile multiplier W_u
# ---------------------------------------------------------------------------

def compute_W_u(es_grid: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """Compute W_u = ES_u(S) / q(u) for all grid points.

    Parameters
    ----------
    es_grid:
        Expected Shortfall values on the grid.
    q_grid:
        Quantile values on the same grid.

    Returns
    -------
    np.ndarray
        W_u values; entries where q(u) = 0 are set to ``np.nan``.
    """
    W = np.empty_like(es_grid, dtype=float)
    nonzero = np.abs(q_grid) > 1e-15
    W[nonzero] = es_grid[nonzero] / q_grid[nonzero]
    W[~nonzero] = np.nan
    return W


# ---------------------------------------------------------------------------
# 5. Randomization parameter theta_u
# ---------------------------------------------------------------------------

def compute_theta_u(
    cdf_s: Callable[[float], float],
    pmf_s: Optional[Callable[[float], float]],
    q_u: float,
    u: float,
) -> float:
    """Compute the randomization parameter theta_u.

    For a *continuous* distribution (``pmf_s=None``) there is no atom at
    q(u) and theta_u = 0.  For a *discrete* distribution::

        theta_u = (F_S(q(u)) - u) / P(S = q(u))

    This is clamped to [0, 1] to guard against floating-point noise.

    Parameters
    ----------
    cdf_s:
        Right-continuous CDF F_S evaluated at a scalar.
    pmf_s:
        Probability mass function P(S = s).  Pass ``None`` for continuous
        distributions.
    q_u:
        Left-continuous quantile q(u).
    u:
        Probability level.

    Returns
    -------
    float
        theta_u in [0, 1].
    """
    if pmf_s is None:
        return 0.0

    p_atom = float(pmf_s(q_u))
    if p_atom < 1e-15:
        return 0.0

    f_at_q = float(cdf_s(q_u))
    theta = (f_at_q - u) / p_atom
    return float(np.clip(theta, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 6. Randomized tail indicator pi_u
# ---------------------------------------------------------------------------

def compute_pi_u(
    s_values: np.ndarray,
    q_u: float,
    theta_u: float,
) -> np.ndarray:
    """Compute the randomized tail indicator pi_u(s).

    pi_u(s) = 1_{s > q_u} + theta_u * 1_{s = q_u}

    For continuous distributions theta_u = 0 and pi_u(s) = 1_{s > q_u}.

    Parameters
    ----------
    s_values:
        Array of aggregate loss values S.
    q_u:
        Quantile q(u).
    theta_u:
        Randomization weight in [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape ``s_values.shape`` with pi_u values in {0, theta_u, 1}.
    """
    s_values = np.asarray(s_values, dtype=float)
    pi = (s_values > q_u).astype(float)
    if theta_u > 0.0:
        pi += theta_u * (s_values == q_u).astype(float)
    return pi


# ---------------------------------------------------------------------------
# 7. Monte Carlo Phi estimator
# ---------------------------------------------------------------------------

def compute_phi_continuous(
    x_i_samples: np.ndarray,
    s_samples: np.ndarray,
    q_u: float,
    theta_u: float,
) -> float:
    """Estimate Phi_{i,u} = E[X_i pi_u(S)] / E[S pi_u(S)] via Monte Carlo.

    This is the proportional Expected Shortfall allocation for component i.

    Parameters
    ----------
    x_i_samples:
        Simulated losses X_i for component i.
    s_samples:
        Simulated aggregate losses S = sum_j X_j.
    q_u:
        Quantile q(u).
    theta_u:
        Randomization weight theta_u.

    Returns
    -------
    float
        Estimated Phi_{i,u}, or ``np.nan`` if the denominator is negligible.
    """
    pi = compute_pi_u(s_samples, q_u, theta_u)
    numerator = float(np.mean(x_i_samples * pi))
    denominator = float(np.mean(s_samples * pi))
    if abs(denominator) < 1e-15:
        return np.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# 8. Monte Carlo Psi estimator
# ---------------------------------------------------------------------------

def compute_psi_continuous(
    r_i_samples: np.ndarray,
    s_samples: np.ndarray,
    q_u: float,
    theta_u: float,
) -> float:
    """Estimate Psi_{i,u} = E[R_i pi_u(S)] / E[pi_u(S)] via Monte Carlo.

    This is the proportional GTE (Generalised Tail Expectation) allocation.
    R_i = X_i / S on {S != 0} and a fixed constant c_i on {S = 0}; see
    ``compute_r_i``.

    Parameters
    ----------
    r_i_samples:
        Simulated conditional shares R_i = X_i / S (pre-computed via
        ``compute_r_i``).
    s_samples:
        Simulated aggregate losses S.
    q_u:
        Quantile q(u).
    theta_u:
        Randomization weight theta_u.

    Returns
    -------
    float
        Estimated Psi_{i,u}, or ``np.nan`` if the denominator is negligible.
    """
    pi = compute_pi_u(s_samples, q_u, theta_u)
    numerator = float(np.mean(r_i_samples * pi))
    denominator = float(np.mean(pi))
    if abs(denominator) < 1e-15:
        return np.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# 9. Conditional share R_i
# ---------------------------------------------------------------------------

def compute_r_i(
    x_i_samples: np.ndarray,
    s_samples: np.ndarray,
    c_i: float = 0.0,
) -> np.ndarray:
    """Compute the conditional share R_i = X_i / S.

    R_i(omega) = X_i(omega) / S(omega)  when S(omega) != 0,
                 c_i                     when S(omega) = 0.

    Parameters
    ----------
    x_i_samples:
        Component losses X_i.
    s_samples:
        Aggregate losses S.
    c_i:
        Fallback value used on the zero-sum event {S = 0}.  Defaults to 0.

    Returns
    -------
    np.ndarray
        Array of R_i values with the same shape as ``x_i_samples``.
    """
    s_samples = np.asarray(s_samples, dtype=float)
    x_i_samples = np.asarray(x_i_samples, dtype=float)
    # Avoid divide-by-zero warning: use a safe denominator, then overwrite zeros.
    s_safe = np.where(s_samples != 0.0, s_samples, 1.0)
    r = x_i_samples / s_safe
    r[s_samples == 0.0] = float(c_i)
    return r


# ---------------------------------------------------------------------------
# 10. Quantile-function Phi
# ---------------------------------------------------------------------------

def compute_phi_quantile(
    q_xi_func: Callable[[float], float],
    q_s_func: Callable[[float], float],
    u: float,
    u_max: float = 1.0 - 1e-8,
    n_points: int = 2000,
) -> float:
    """Compute Phi_{i,u} via quantile-function quadrature.

    Phi_{i,u} = integral_u^1 q_{X_i}(v) dv  /  integral_u^1 q_S(v) dv

    The integrals are approximated on a dense uniform grid [u, u_max],
    where u_max < 1 avoids numerical divergence near the right tail.  The
    factor (1-u) cancels in the ratio.

    Parameters
    ----------
    q_xi_func:
        Quantile function of component X_i, callable on a single float.
    q_s_func:
        Quantile function of aggregate S, callable on a single float.
    u:
        Lower probability level.
    u_max:
        Upper truncation level (default 1 - 1e-8).
    n_points:
        Number of quadrature nodes.

    Returns
    -------
    float
        Phi_{i,u}, or ``np.nan`` if the denominator is negligible.
    """
    if u >= u_max:
        return np.nan

    v_grid = np.linspace(u, u_max, n_points)
    q_xi_vals = np.array([q_xi_func(v) for v in v_grid], dtype=float)
    q_s_vals = np.array([q_s_func(v) for v in v_grid], dtype=float)

    num = np.trapz(q_xi_vals, v_grid)
    den = np.trapz(q_s_vals, v_grid)

    if abs(den) < 1e-15:
        return np.nan
    return float(num / den)


# ---------------------------------------------------------------------------
# 11. Quantile-function Psi
# ---------------------------------------------------------------------------

def compute_psi_quantile(
    q_xi_func: Callable[[float], float],
    q_s_func: Callable[[float], float],
    u: float,
    u_max: float = 1.0 - 1e-8,
    n_points: int = 2000,
) -> float:
    """Compute Psi_{i,u} via quantile-function quadrature.

    Psi_{i,u} = (1/(1-u)) * integral_u^1 q_{X_i}(v) / q_S(v) dv
              = (1/(1-u)) * integral_u^1 h(v) dv

    Grid points where q_S(v) = 0 are excluded from the trapezoidal sum.

    Parameters
    ----------
    q_xi_func:
        Quantile function of component X_i.
    q_s_func:
        Quantile function of aggregate S.
    u:
        Lower probability level.
    u_max:
        Upper truncation level (default 1 - 1e-8).
    n_points:
        Number of quadrature nodes.

    Returns
    -------
    float
        Psi_{i,u}, or ``np.nan`` if the result is undefined.
    """
    if u >= u_max:
        return np.nan

    denom_width = 1.0 - u
    if denom_width < 1e-15:
        return np.nan

    v_grid = np.linspace(u, u_max, n_points)
    q_s_vals = np.array([q_s_func(v) for v in v_grid], dtype=float)
    q_xi_vals = np.array([q_xi_func(v) for v in v_grid], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h_vals = np.where(np.abs(q_s_vals) > 1e-15, q_xi_vals / q_s_vals, np.nan)

    valid = ~np.isnan(h_vals)
    if not np.any(valid):
        return np.nan

    # Trapezoidal integral of h over valid sub-grid; normalise by (1-u).
    integral = np.trapz(h_vals[valid], v_grid[valid])
    return float(integral / denom_width)


# ---------------------------------------------------------------------------
# 12. Pointwise conditional quantile ratio h(u)
# ---------------------------------------------------------------------------

def compute_h(
    q_xi_func: Callable[[float], float],
    q_s_func: Callable[[float], float],
    u_grid: np.ndarray,
) -> np.ndarray:
    """Compute h(u) = q_{X_i}(u) / q_S(u) on the grid.

    Parameters
    ----------
    q_xi_func:
        Quantile function of component X_i.
    q_s_func:
        Quantile function of aggregate S.
    u_grid:
        Array of probability levels.

    Returns
    -------
    np.ndarray
        Array of h(u) values; entries where q_S(u) = 0 are ``np.nan``.
    """
    h = np.empty(len(u_grid), dtype=float)
    for i, u in enumerate(u_grid):
        qs = float(q_s_func(u))
        if abs(qs) < 1e-15:
            h[i] = np.nan
        else:
            h[i] = float(q_xi_func(u)) / qs
    return h


# ---------------------------------------------------------------------------
# 13. Covariance identity verification
# ---------------------------------------------------------------------------

def compute_covariance_identity(
    q_s_func: Callable[[float], float],
    h_func: Callable[[float], float],
    u: float,
    u_max: float = 1.0 - 1e-8,
    n_points: int = 5000,
) -> Dict[str, float]:
    """Verify the covariance identity for the allocation gap.

    The identity states:

        Phi_{i,u} - Psi_{i,u} = Cov_u(q_S(V), h(V)) / E_u[q_S(V)]

    where V ~ Uniform(u, 1) and h(v) = q_{X_i}(v) / q_S(v).

    Proof sketch
    ------------
    Let the expectation under V ~ U(u, u_max) be E_u[f] = (1/w) integral f dv
    with w = u_max - u.  Then::

        E_u[q_S(V) * h(V)] = (1/w) integral q_S(v) h(v) dv
                            = (1/w) integral q_{X_i}(v) dv
                            = E_u[q_{X_i}(V)]

        Cov_u(q_S, h) = E_u[q_S * h] - E_u[q_S] * E_u[h]
                      = E_u[q_{X_i}] - E_u[q_S] * Psi_{i,u}

        Cov_u / E_u[q_S] = E_u[q_{X_i}] / E_u[q_S] - Psi_{i,u}
                         = Phi_{i,u} - Psi_{i,u}   ✓

    Parameters
    ----------
    q_s_func:
        Quantile function q_S(v) of the aggregate.
    h_func:
        Ratio function h(v) = q_{X_i}(v) / q_S(v).
    u:
        Lower probability level.
    u_max:
        Upper truncation (default 1 - 1e-8).
    n_points:
        Number of quadrature nodes.

    Returns
    -------
    Dict[str, float]
        ``'cov_term'``       – Cov_u(q_S(V), h(V))
        ``'delta_from_cov'`` – Cov_u / E_u[q_S] = Phi_{i,u} - Psi_{i,u}
        ``'eu_q'``           – E_u[q_S(V)]  (= ES_u(S) up to u_max approximation)
    """
    if u >= u_max:
        return {"cov_term": np.nan, "delta_from_cov": np.nan, "eu_q": np.nan}

    width = u_max - u
    if width < 1e-15:
        return {"cov_term": np.nan, "delta_from_cov": np.nan, "eu_q": np.nan}

    v_grid = np.linspace(u, u_max, n_points)
    q_vals = np.array([q_s_func(v) for v in v_grid], dtype=float)
    h_vals = np.array([h_func(v) for v in v_grid], dtype=float)

    # E_u[q_S(V)] computed over the full grid (q is always finite).
    eu_q = np.trapz(q_vals, v_grid) / width

    # Masks for finite h values.
    valid = np.isfinite(h_vals)
    if not np.any(valid):
        return {"cov_term": np.nan, "delta_from_cov": np.nan, "eu_q": float(eu_q)}

    # E_u[h(V)] – integrate over valid sub-grid, normalise by full width.
    eu_h = np.trapz(h_vals[valid], v_grid[valid]) / width

    # E_u[q_S(V) * h(V)] – product; use valid points only.
    qh_vals = q_vals * h_vals
    eu_qh = np.trapz(qh_vals[valid], v_grid[valid]) / width

    # Covariance under U(u, u_max).
    cov = eu_qh - eu_q * eu_h

    if abs(eu_q) < 1e-15:
        delta_from_cov = np.nan
    else:
        delta_from_cov = cov / eu_q

    return {
        "cov_term": float(cov),
        "delta_from_cov": float(delta_from_cov),
        "eu_q": float(eu_q),
    }


# ---------------------------------------------------------------------------
# 14. Effective tail domain
# ---------------------------------------------------------------------------

def compute_effective_tail_domain(
    cdf_s: Callable[[float], float],
    q_u_star: float,
    u_star: float,
    s_max: float,
) -> Dict:
    """Determine the effective tail domain at level u*.

    The effective tail domain is the support of pi_{u*}(s):

    * **Open** at the lower boundary: F_S(q(u*)) = u*.
      There is no genuine atom at q(u*) (continuous case or zero-mass atom),
      so the domain is the open interval (q(u*), s_max].

    * **Closed** at the lower boundary: F_S(q(u*)) > u*.
      There is a genuine atom at q(u*) with positive mass, so theta_{u*} > 0
      and the domain is the closed interval [q(u*), s_max].

    Parameters
    ----------
    cdf_s:
        Right-continuous CDF F_S.
    q_u_star:
        Quantile q(u*).
    u_star:
        Probability level u*.
    s_max:
        Upper support bound.

    Returns
    -------
    Dict
        ``'lower_open'`` (bool)  – True iff F_S(q(u*)) = u* (open at lower end)
        ``'lower'``      (float) – lower boundary = q(u*)
        ``'upper'``      (float) – upper boundary = s_max
    """
    f_at_q = float(cdf_s(q_u_star))
    # Open iff F_S(q(u*)) == u* (no genuine atom, continuous case).
    lower_open = abs(f_at_q - float(u_star)) < 1e-10
    return {
        "lower_open": lower_open,
        "lower": float(q_u_star),
        "upper": float(s_max),
    }


# ---------------------------------------------------------------------------
# 15. Allocation validation
# ---------------------------------------------------------------------------

def validate_allocations(
    phi_list: List[float],
    psi_list: List[float],
    tol: float = 1e-6,
) -> Dict:
    """Check that allocation weights sum to one.

    By the budget identity, sum_i Phi_{i,u} = 1 and sum_i Psi_{i,u} = 1
    for all u whenever the components X_i are non-negative and S = sum_i X_i.

    Parameters
    ----------
    phi_list:
        Sequence of Phi_{i,u} values for i = 1, …, n.
    psi_list:
        Sequence of Psi_{i,u} values for i = 1, …, n.
    tol:
        Absolute deviation from 1 considered acceptable.

    Returns
    -------
    Dict
        ``'phi_sum'`` – sum of Phi values
        ``'psi_sum'`` – sum of Psi values
        ``'phi_ok'``  – True if |sum - 1| < tol
        ``'psi_ok'``  – True if |sum - 1| < tol
    """
    phi_arr = np.asarray(phi_list, dtype=float)
    psi_arr = np.asarray(psi_list, dtype=float)

    phi_sum = float(np.nansum(phi_arr))
    psi_sum = float(np.nansum(psi_arr))

    return {
        "phi_sum": phi_sum,
        "psi_sum": psi_sum,
        "phi_ok": abs(phi_sum - 1.0) < tol,
        "psi_ok": abs(psi_sum - 1.0) < tol,
    }


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class TailAllocationScenario:
    """Container for a computed tail-allocation scenario.

    Attributes
    ----------
    name:
        Descriptive label for the scenario.
    u_grid:
        Probability levels at which quantities are evaluated.
    q_grid:
        Quantile values q(u) on ``u_grid``.
    es_grid:
        Expected Shortfall ES_u(S) on ``u_grid``.
    W_grid:
        Tail multiplier W_u = ES_u / q(u) on ``u_grid``.
    phi_results:
        Dict mapping component name → array of Phi_{i,u} over ``u_grid``.
    psi_results:
        Dict mapping component name → array of Psi_{i,u} over ``u_grid``.
    delta_results:
        Dict mapping component name → array of (Phi_{i,u} - Psi_{i,u}).
    h_results:
        Dict mapping component name → array of h_i(u) = q_{X_i}(u)/q_S(u).
    """

    name: str
    u_grid: np.ndarray
    q_grid: np.ndarray
    es_grid: np.ndarray
    W_grid: np.ndarray
    phi_results: Dict[str, np.ndarray] = field(default_factory=dict)
    psi_results: Dict[str, np.ndarray] = field(default_factory=dict)
    delta_results: Dict[str, np.ndarray] = field(default_factory=dict)
    h_results: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy ``pd.DataFrame`` with all scenario quantities.

        Columns
        -------
        u, q, es, W           – Core risk measures on the grid.
        phi_{name}            – Phi allocation for each component.
        psi_{name}            – Psi allocation for each component.
        delta_{name}          – Phi - Psi gap for each component.
        h_{name}              – Quantile ratio h_i(u) for each component.

        Returns
        -------
        pd.DataFrame
            One row per grid point.
        """
        df = pd.DataFrame(
            {
                "u": self.u_grid,
                "q": self.q_grid,
                "es": self.es_grid,
                "W": self.W_grid,
            }
        )

        for comp, vals in self.phi_results.items():
            df[f"phi_{comp}"] = np.asarray(vals, dtype=float)

        for comp, vals in self.psi_results.items():
            df[f"psi_{comp}"] = np.asarray(vals, dtype=float)

        for comp, vals in self.delta_results.items():
            df[f"delta_{comp}"] = np.asarray(vals, dtype=float)

        for comp, vals in self.h_results.items():
            df[f"h_{comp}"] = np.asarray(vals, dtype=float)

        return df


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("tail_allocation.py — self-test with N(0,1) + N(0,1) scenario")
    print("=" * 60)

    # Aggregate S = X1 + X2, X1, X2 ~ N(0, 1) => S ~ N(0, sqrt(2))
    sigma_s = np.sqrt(2.0)
    cdf_s = lambda x: stats.norm.cdf(x, scale=sigma_s)
    q_s = lambda u: stats.norm.ppf(u, scale=sigma_s)
    q_x1 = lambda u: stats.norm.ppf(u, scale=1.0)

    u_grid = np.linspace(0.50, 0.99, 50)
    q_grid = compute_quantile_grid(cdf_s, u_grid, x_min=-20.0, x_max=20.0)
    es_grid = compute_es(q_grid, u_grid)
    W_grid = compute_W_u(es_grid, q_grid)

    # Phi and Psi via quantile quadrature.
    phi_vals = np.array([
        compute_phi_quantile(q_x1, q_s, u) for u in u_grid
    ])
    psi_vals = np.array([
        compute_psi_quantile(q_x1, q_s, u) for u in u_grid
    ])
    delta_vals = phi_vals - psi_vals
    h_vals = compute_h(q_x1, q_s, u_grid)

    # Assemble scenario.
    scenario = TailAllocationScenario(
        name="Normal(0,1) + Normal(0,1)",
        u_grid=u_grid,
        q_grid=q_grid,
        es_grid=es_grid,
        W_grid=W_grid,
        phi_results={"X1": phi_vals},
        psi_results={"X1": psi_vals},
        delta_results={"X1": delta_vals},
        h_results={"X1": h_vals},
    )

    df = scenario.to_dataframe()
    print(df[["u", "q", "es", "W", "phi_X1", "psi_X1", "delta_X1"]].tail(10).to_string(index=False))

    # Validate: by symmetry Phi = Psi = 0.5 for both components.
    check = validate_allocations([phi_vals[-1], 1.0 - phi_vals[-1]],
                                 [psi_vals[-1], 1.0 - psi_vals[-1]])
    print(f"\nValidation at u = {u_grid[-1]:.2f}: {check}")

    # Covariance identity check.
    h_func = lambda v: q_x1(v) / q_s(v) if abs(q_s(v)) > 1e-12 else np.nan
    cov_info = compute_covariance_identity(q_s, h_func, u=0.95)
    print(f"\nCovariance identity at u=0.95: {cov_info}")
    print(f"Direct delta at u=0.95:        {np.interp(0.95, u_grid, delta_vals):.6f}")

    print("\nDone.")
