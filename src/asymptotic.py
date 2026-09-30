"""
asymptotic.py
=============
Asymptotic tail-equivalence analysis and counterexamples.

Studies the limiting behavior of allocation measures Phi_{i,u} and Psi_{i,u}
as u -> 1 through three canonical scenarios:

  Scenario 1 — Bounded W_u (Pareto, comonotonic):
    W_u stays bounded; Phi and Psi share a common limit determined by the
    heaviest-tailed component(s).

  Scenario 2 — W_u -> 1 with oscillating share (Exponential aggregate):
    W_u -> 1 but g_i(s) oscillates, causing both Phi and Psi to oscillate
    without a common limit (delta = Phi - Psi does vanish).

  Scenario 3 — Discrete alternating-share (diverging-W spirit):
    Discrete S with rapidly growing atoms and an alternating conditional share;
    Phi converges to 1/2 while Psi oscillates between two limiting values.

Key objects
-----------
compute_W_sequence             Evaluate W_u = ES_u(S)/q_u along a u-sequence.
compute_phi_psi_sequence       Evaluate Phi and Psi along a u-sequence.
estimate_convergence           limsup/liminf/oscillation diagnostics.
check_theorem_bounds           Verify |Psi - Phi| <= 2*(W_u-1)*sup|Phi|.
ParetoBoundedWScenario         Analytic Pareto comonotonic scenario.
OscillatingShareScenario       Exp(1) aggregate with oscillating conditional share.
DiscreteDivergentWScenario     Discrete S with alternating component shares.
compute_asymptotic_diagnostics Tidy DataFrame of all sequence diagnostics.
classify_asymptotic_behavior   String label for the asymptotic regime.
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize
from typing import Callable, Tuple, Dict, List, Optional
import warnings


# ---------------------------------------------------------------------------
# 1. W_u sequence
# ---------------------------------------------------------------------------

def compute_W_sequence(
    q_func: Callable[[float], float],
    es_func: Callable[[float], float],
    u_seq: np.ndarray,
) -> np.ndarray:
    """Evaluate W_u = ES_u(S) / q_u along a sequence of levels.

    Parameters
    ----------
    q_func:
        Quantile function q_S: [0, 1) -> R.
    es_func:
        Expected-shortfall function ES_u(S): [0, 1) -> R.
    u_seq:
        Strictly increasing 1-D array of probability levels approaching 1.

    Returns
    -------
    np.ndarray
        W_u values; entries where |q_u| < 1e-15 are set to ``np.nan``.
    """
    q_vals = np.array([q_func(u) for u in u_seq], dtype=float)
    es_vals = np.array([es_func(u) for u in u_seq], dtype=float)
    safe = np.abs(q_vals) > 1e-15
    q_safe = np.where(safe, q_vals, 1.0)   # replace 0s to avoid divide-by-zero warning
    W = np.where(safe, es_vals / q_safe, np.nan)
    return W


# ---------------------------------------------------------------------------
# 2. Phi / Psi sequences
# ---------------------------------------------------------------------------

def compute_phi_psi_sequence(
    phi_func: Callable[[float], float],
    psi_func: Callable[[float], float],
    u_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate Phi and Psi along a sequence u_k -> 1.

    Parameters
    ----------
    phi_func:
        Callable returning Phi_{i,u} for a given u.
    psi_func:
        Callable returning Psi_{i,u} for a given u.
    u_seq:
        Strictly increasing 1-D array of probability levels approaching 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (phi_seq, psi_seq) — arrays of the same length as ``u_seq``.
    """
    phi_seq = np.array([phi_func(u) for u in u_seq], dtype=float)
    psi_seq = np.array([psi_func(u) for u in u_seq], dtype=float)
    return phi_seq, psi_seq


# ---------------------------------------------------------------------------
# 3. Convergence diagnostics
# ---------------------------------------------------------------------------

def estimate_convergence(values: np.ndarray, window: int = 10) -> Dict:
    """Estimate asymptotic convergence using the last ``window`` finite values.

    The function inspects the tail of ``values``, discards NaN entries, and
    returns limsup / liminf / oscillation amplitude over that tail.

    Parameters
    ----------
    values:
        1-D array whose entries represent a sequence indexed by a u-grid.
    window:
        Number of trailing values to inspect.

    Returns
    -------
    Dict with keys:
        ``'limsup'``               – max of the finite tail values.
        ``'liminf'``               – min of the finite tail values.
        ``'oscillation_amplitude'`` – limsup − liminf.
        ``'converged'``            – True iff oscillation_amplitude < 1e-4.
    """
    tail = values[-window:]
    valid = tail[~np.isnan(tail)]

    if len(valid) == 0:
        return {
            "limsup": np.nan,
            "liminf": np.nan,
            "oscillation_amplitude": np.nan,
            "converged": False,
        }

    limsup = float(np.max(valid))
    liminf = float(np.min(valid))
    osc = limsup - liminf
    # Threshold 0.03: coarser than machine epsilon but well below the oscillation
    # amplitudes seen in the non-converging scenarios (≥ 0.06) while still
    # flagging slowly-drifting monotone tails (amplitude < 0.03) as settled.
    return {
        "limsup": limsup,
        "liminf": liminf,
        "oscillation_amplitude": float(osc),
        "converged": bool(osc < 0.03),
    }


# ---------------------------------------------------------------------------
# 4. Theorem-bound checker
# ---------------------------------------------------------------------------

def check_theorem_bounds(
    phi_seq: np.ndarray,
    psi_seq: np.ndarray,
    W_seq: np.ndarray,
    tol: float = 1e-8,
) -> Dict:
    """Verify the theoretical bound |Psi - Phi| <= 2*(W_u - 1)*sup|Phi| pointwise.

    The bound reflects the general inequality proved in the tail-allocation
    framework: the discrepancy between the two allocation measures is controlled
    by how far W_u is from 1 (i.e., how heavy the tail of S is relative to its
    quantile) and by the scale of Phi itself.

    Parameters
    ----------
    phi_seq:
        Array of Phi_{i,u} values along the sequence.
    psi_seq:
        Array of Psi_{i,u} values along the sequence.
    W_seq:
        Array of W_u values along the sequence.
    tol:
        Absolute tolerance; a point is considered a violation only if the
        residual (bound − |Psi − Phi|) is more negative than ``−tol``.

    Returns
    -------
    Dict with keys:
        ``'bound_holds'``    – bool array; True where the bound is satisfied.
        ``'residuals'``      – float array; bound − |Psi − Phi| (positive = slack).
        ``'max_violation'``  – largest violation magnitude (0.0 if none).
    """
    sup_phi = float(np.nanmax(np.abs(phi_seq)))
    lhs = np.abs(psi_seq - phi_seq)
    rhs = 2.0 * np.maximum(W_seq - 1.0, 0.0) * sup_phi

    residuals = rhs - lhs
    bound_holds = np.where(np.isnan(residuals), False, residuals >= -tol)
    violations = np.where(~np.isnan(residuals), np.maximum(-residuals, 0.0), 0.0)
    max_violation = float(np.max(violations)) if violations.size > 0 else 0.0

    return {
        "bound_holds": bound_holds,
        "residuals": residuals,
        "max_violation": max_violation,
    }


# ---------------------------------------------------------------------------
# 5. Scenario 1 — Pareto, bounded W_u
# ---------------------------------------------------------------------------

class ParetoBoundedWScenario:
    """Comonotonic Pareto scenario where W_u remains bounded as u -> 1.

    Each component has a Pareto quantile function
      q_{X_i}(u) = sigma_i * (1 - u)^{-1/alpha_i},  alpha_i > 1.

    The comonotonic aggregate is
      q_S(u) = sum_i sigma_i * (1 - u)^{-1/alpha_i}.

    The ES is available in closed form:
      ES_u(S) = sum_i sigma_i * alpha_i/(alpha_i - 1) * (1 - u)^{-1/alpha_i},

    so W_u = ES_u(S)/q_S(u) is a ratio of weighted sums of the same power
    factors and stays bounded (converges to alpha_min/(alpha_min - 1) in the
    single-component case, or to a sigma-weighted average when all alphas are
    equal).

    As u -> 1 the allocations converge to
      ell_i = sigma_i / sum_{j: alpha_j = alpha_min} sigma_j   if alpha_i = alpha_min,
      ell_i = 0                                                  otherwise,
    for both Phi and Psi.

    Parameters
    ----------
    alphas:
        Tail indices alpha_i > 1, one per component.
    sigmas:
        Scale parameters sigma_i > 0, one per component.
    """

    def __init__(self, alphas: List[float], sigmas: List[float]) -> None:
        if any(a <= 1.0 for a in alphas):
            raise ValueError("All alphas must be > 1 for ES to exist.")
        if any(s <= 0.0 for s in sigmas):
            raise ValueError("All sigmas must be > 0.")
        self.alphas: List[float] = list(alphas)
        self.sigmas: List[float] = list(sigmas)

    def q_s(self, u: float) -> float:
        """Comonotonic aggregate quantile q_S(u) = sum_i sigma_i*(1-u)^{-1/alpha_i}."""
        eps = 1.0 - u
        if eps < 1e-15:
            return np.nan
        return sum(s * eps ** (-1.0 / a) for s, a in zip(self.sigmas, self.alphas))

    def es_s(self, u: float) -> float:
        """Expected Shortfall ES_u(S) = sum_i sigma_i * alpha_i/(alpha_i-1) * (1-u)^{-1/alpha_i}.

        Derivation: int_u^1 (1-v)^{-1/a} dv = a/(a-1) * (1-u)^{1-1/a},  a > 1,
        so ES_u = (1/(1-u)) * sum_i sigma_i * a_i/(a_i-1) * (1-u)^{1-1/a_i}
                = sum_i sigma_i * a_i/(a_i-1) * (1-u)^{-1/a_i}.
        """
        eps = 1.0 - u
        if eps < 1e-15:
            return np.nan
        return sum(
            s * a / (a - 1.0) * eps ** (-1.0 / a)
            for s, a in zip(self.sigmas, self.alphas)
        )

    def W_u(self, u: float) -> float:
        """Tail multiplier W_u = ES_u(S) / q_S(u)."""
        q = self.q_s(u)
        es = self.es_s(u)
        if q is np.nan or np.isnan(q) or abs(q) < 1e-300:
            return np.nan
        return es / q

    def phi_i(self, i: int, u: float) -> float:
        """Phi_{i,u} = int_u^1 q_{X_i}(v) dv / int_u^1 q_S(v) dv.

        Analytic closed form using int_u^1 (1-v)^{-1/a} dv = a/(a-1)*(1-u)^{1-1/a}:

          Phi_{i,u} = sigma_i * alpha_i/(alpha_i-1) * (1-u)^{1-1/alpha_i}
                      / sum_j sigma_j * alpha_j/(alpha_j-1) * (1-u)^{1-1/alpha_j}.
        """
        eps = 1.0 - u
        if eps < 1e-15:
            return np.nan
        numerator = (
            self.sigmas[i] * self.alphas[i] / (self.alphas[i] - 1.0)
            * eps ** (1.0 - 1.0 / self.alphas[i])
        )
        denominator = sum(
            s * a / (a - 1.0) * eps ** (1.0 - 1.0 / a)
            for s, a in zip(self.sigmas, self.alphas)
        )
        if abs(denominator) < 1e-300:
            return np.nan
        return numerator / denominator

    def psi_i(self, i: int, u: float) -> float:
        """Psi_{i,u} = (1/(1-u)) * int_u^1 q_{X_i}(v)/q_S(v) dv.

        Computed via the substitution t = 1 - v, which maps the integral to

          (1/(1-u)) * int_0^{1-u} sigma_i * t^{-1/alpha_i}
                                   / sum_j sigma_j * t^{-1/alpha_j} dt.

        The integrand is numerically stabilised by factoring out
        t^{-1/alpha_min}, converting all power factors to non-negative
        exponents t^{1/alpha_min - 1/alpha_j} in [0, 1] for t in (0, 1].
        """
        eps = 1.0 - u
        if eps < 1e-15:
            return np.nan

        alpha_min = min(self.alphas)

        def share(t: float) -> float:
            # Factor out t^{-1/alpha_min}; remaining exponents are >= 0.
            exp_i = 1.0 / alpha_min - 1.0 / self.alphas[i]
            num = self.sigmas[i] * (t ** exp_i if t > 0.0 else (1.0 if exp_i == 0.0 else 0.0))
            denom = sum(
                s * (t ** (1.0 / alpha_min - 1.0 / a) if t > 0.0
                     else (1.0 if abs(1.0 / alpha_min - 1.0 / a) < 1e-14 else 0.0))
                for s, a in zip(self.sigmas, self.alphas)
            )
            return num / denom if denom > 1e-300 else 0.0

        result, _ = integrate.quad(
            share, 0.0, eps, limit=200, epsabs=1e-10, epsrel=1e-8
        )
        return result / eps

    def asymptotic_limit(self, i: int) -> float:
        """Limiting allocation ell_i as u -> 1.

        The component(s) with the heaviest tail (smallest alpha) dominate:

          ell_i = sigma_i / sum_{j: alpha_j = alpha_min} sigma_j   if alpha_i = alpha_min,
          ell_i = 0                                                  otherwise.
        """
        alpha_min = min(self.alphas)
        if abs(self.alphas[i] - alpha_min) > 1e-10:
            return 0.0
        denom = sum(
            s for s, a in zip(self.sigmas, self.alphas)
            if abs(a - alpha_min) < 1e-10
        )
        return self.sigmas[i] / denom


# ---------------------------------------------------------------------------
# 6. Scenario 2 — Oscillating share, W_u -> 1
# ---------------------------------------------------------------------------

class OscillatingShareScenario:
    """Exponential aggregate with an oscillating conditional share.

    S ~ Exp(1) has quantile q_S(u) = -log(1-u) and ES_u(S) = q_S(u) + 1,
    giving W_u = 1 + 1/q_S(u) -> 1 as u -> 1.

    Component 1 is defined by the conditional share
      g_1(s) = 0.5 + amplitude * sin(frequency * s),
    so X_1 = g_1(S)*S.  Component 2 is X_2 = (1 - g_1(S))*S.

    Both Phi and Psi oscillate as u -> 1 because g_1 oscillates along the
    growing quantile q_S(u), while their difference Delta = Phi - Psi -> 0
    (since W_u -> 1).

    Parameters
    ----------
    amplitude:
        Oscillation amplitude A of the share function; must satisfy
        |A| < 0.5 to keep 0 < g_1(s) < 1.
    frequency:
        Angular frequency omega (radians per unit of s).
    """

    def __init__(
        self, amplitude: float = 0.3, frequency: float = 2.0 * np.pi
    ) -> None:
        if abs(amplitude) >= 0.5:
            raise ValueError("|amplitude| must be < 0.5 to keep g_i in (0, 1).")
        self.amplitude = amplitude
        self.frequency = frequency

    def g_i(self, s: float) -> float:
        """Oscillating conditional share for component 1: g_1(s) = 0.5 + A*sin(omega*s)."""
        return 0.5 + self.amplitude * np.sin(self.frequency * s)

    def q_s(self, u: float) -> float:
        """Quantile of S ~ Exp(1): q_S(u) = -log(1-u)."""
        if u >= 1.0 - 1e-15:
            return np.inf
        return -np.log(1.0 - u)

    def es_s(self, u: float) -> float:
        """Expected Shortfall of S ~ Exp(1): ES_u(S) = -log(1-u) + 1.

        Derivation: int_u^1 (-log(1-v)) dv
        (sub t = 1-v) = int_0^{1-u} (-log t) dt = [(1-u)*(-log(1-u)) + (1-u)],
        so ES_u = (1-u)*(-log(1-u)) + (1-u)] / (1-u) = -log(1-u) + 1 = q_S(u) + 1.
        """
        return self.q_s(u) + 1.0

    def W_u(self, u: float) -> float:
        """Tail multiplier W_u = 1 + 1/q_S(u) -> 1 as u -> 1."""
        q = self.q_s(u)
        if np.isinf(q) or abs(q) < 1e-15:
            return np.nan
        return (q + 1.0) / q

    def phi_i(self, u: float, n_quad: int = 10000) -> float:
        """Phi_{1,u} = int_{q_u}^inf g_1(s)*s*exp(-s) ds / int_{q_u}^inf s*exp(-s) ds.

        The denominator equals (q_u + 1)*exp(-q_u) analytically; the numerator
        is evaluated with ``scipy.integrate.quad`` to handle oscillations.

        Parameters
        ----------
        u:
            Probability level in [0, 1).
        n_quad:
            Unused (kept for interface compatibility); ``integrate.quad`` is used.

        Returns
        -------
        float
            Phi_{1,u}; ``np.nan`` if u is too close to 1 or denominator is tiny.
        """
        if u >= 1.0 - 1e-10:
            return np.nan
        q_u = self.q_s(u)
        # Analytic denominator: int_{q_u}^inf s*exp(-s) ds = (q_u + 1)*exp(-q_u)
        denom = (q_u + 1.0) * np.exp(-q_u)
        if abs(denom) < 1e-20:
            return np.nan
        num, _ = integrate.quad(
            lambda s: self.g_i(s) * s * np.exp(-s),
            q_u,
            np.inf,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-8,
        )
        return num / denom

    def psi_i(self, u: float) -> float:
        """Psi_{1,u} = int_{q_u}^inf g_1(s)*exp(-s) ds / exp(-q_u).

        The denominator int_{q_u}^inf exp(-s) ds = exp(-q_u) is exact.
        The oscillating integral in the numerator can be resolved analytically:

          int_{q_u}^inf sin(omega*s)*exp(-s) ds
          = exp(-q_u) * (omega*cos(omega*q_u) + sin(omega*q_u)) / (1 + omega^2),

        giving Psi_{1,u} = 0.5 + A*(omega*cos(omega*q_u) + sin(omega*q_u)) / (1 + omega^2).

        This analytic form is used for accuracy and speed.

        Parameters
        ----------
        u:
            Probability level in [0, 1).

        Returns
        -------
        float
            Psi_{1,u}; ``np.nan`` if u is too close to 1.
        """
        if u >= 1.0 - 1e-10:
            return np.nan
        q_u = self.q_s(u)
        omega = self.frequency
        # Analytic result from the Laplace-transform integral formula.
        oscillating_part = (
            omega * np.cos(omega * q_u) + np.sin(omega * q_u)
        ) / (1.0 + omega ** 2)
        return 0.5 + self.amplitude * oscillating_part


# ---------------------------------------------------------------------------
# 7. Scenario 3 — Discrete, divergent-W spirit
# ---------------------------------------------------------------------------

class DiscreteDivergentWScenario:
    """Discrete S with rapidly growing atoms and alternating conditional shares.

    Construction
    ------------
    S takes values {2^k : k = 0, 1, …, N} with probabilities
      p_k = 0.5^{k+1}  for k < N,
      p_N  adjusted so that the total mass equals 1.

    Component 1 is defined by X_1 = alpha_k * S on the atom {S = 2^k},
    where alpha_k alternates between alpha_even (k even) and alpha_odd (k odd).

    This construction causes:
    * Phi_{1,u} to converge to (alpha_even + alpha_odd)/2 = 0.5 (for the
      default parameters) as u -> 1, because the atoms' S-weighted average of
      alpha_k converges (equal mass spacing in S cancels odd/even difference).
    * Psi_{1,u} to oscillate between two limiting values determined by the
      geometric-series weights, because the probability-weighted average of
      alpha_k depends on the parity of the atom at the tail threshold.

    The randomized tail indicator pi_u uses the exact theta_u formula:
      theta_u = (F_S(q_u) - u) / P(S = q_u),  clamped to [0, 1].

    Parameters
    ----------
    N:
        Number of atoms is N + 1; atom k has value 2^k.
    alpha_even:
        Conditional share for even-indexed atoms.
    alpha_odd:
        Conditional share for odd-indexed atoms.
    """

    def __init__(
        self,
        N: int = 30,
        alpha_even: float = 0.3,
        alpha_odd: float = 0.7,
    ) -> None:
        self.N = N
        self.s_vals = np.array([2 ** k for k in range(N + 1)], dtype=float)
        raw_probs = np.array([0.5 ** (k + 1) for k in range(N + 1)], dtype=float)
        # Ensure exact normalisation by adding residual mass to the last atom.
        raw_probs[-1] += 1.0 - raw_probs.sum()
        self.probs = raw_probs
        self.alpha_vals = np.array(
            [alpha_even if k % 2 == 0 else alpha_odd for k in range(N + 1)],
            dtype=float,
        )

    def cdf_s(self, s: float) -> float:
        """Right-continuous CDF F_S(s) = P(S <= s)."""
        return float(np.sum(self.probs[self.s_vals <= s]))

    def q_s(self, u: float) -> float:
        """Left-continuous quantile q_S(u) = inf{s : F_S(s) >= u}.

        Uses ``np.searchsorted`` on the cumulative probability vector to
        locate the atom index k* such that F_S(2^{k*}) >= u > F_S(2^{k*-1}).
        """
        cumprob = np.cumsum(self.probs)
        idx = int(np.searchsorted(cumprob, u, side="left"))
        idx = min(idx, self.N)
        return float(self.s_vals[idx])

    def _theta_u(self, q_u: float, u: float) -> float:
        """Compute theta_u = (F_S(q_u) - u) / P(S = q_u), clamped to [0, 1].

        Parameters
        ----------
        q_u:
            Quantile value q_S(u).
        u:
            Probability level.

        Returns
        -------
        float
            theta_u in [0, 1].
        """
        mask_at = np.abs(self.s_vals - q_u) < 1e-10
        p_at = float(self.probs[mask_at].sum()) if mask_at.any() else 0.0
        if p_at < 1e-300:
            return 0.0
        f_at_q = self.cdf_s(q_u)
        theta = (f_at_q - u) / p_at
        return float(np.clip(theta, 0.0, 1.0))

    def es_s(self, u: float) -> float:
        """Expected Shortfall ES_u(S) = E[S * pi_u(S)] / E[pi_u(S)].

        The randomized tail indicator pi_u(s) = 1_{s > q_u} + theta_u*1_{s = q_u}
        handles the discrete atoms at the threshold exactly.
        """
        q_u = self.q_s(u)
        theta_u = self._theta_u(q_u, u)

        mask_above = self.s_vals > q_u
        mask_at = np.abs(self.s_vals - q_u) < 1e-10

        p_at = float(self.probs[mask_at].sum()) if mask_at.any() else 0.0

        num = (
            float(np.sum(self.s_vals[mask_above] * self.probs[mask_above]))
            + theta_u * q_u * p_at
        )
        denom = (
            float(np.sum(self.probs[mask_above])) + theta_u * p_at
        )
        if denom < 1e-300:
            return np.nan
        return num / denom

    def W_u(self, u: float) -> float:
        """Tail multiplier W_u = ES_u(S) / q_S(u)."""
        q_u = self.q_s(u)
        es = self.es_s(u)
        if np.isnan(es) or abs(q_u) < 1e-300:
            return np.nan
        return es / q_u

    def phi_i(self, u: float) -> float:
        """Phi_{1,u} = E[alpha(S)*S*pi_u(S)] / E[S*pi_u(S)].

        Since X_1 = alpha_k * S on atom k, we have
          Phi_{1,u} = sum_{k: s_k > q_u} alpha_k * s_k * p_k + theta_u * alpha_{k*} * q_u * p_{k*}
                    / sum_{k: s_k > q_u} s_k * p_k + theta_u * q_u * p_{k*}.
        """
        q_u = self.q_s(u)
        theta_u = self._theta_u(q_u, u)

        mask_above = self.s_vals > q_u
        mask_at = np.abs(self.s_vals - q_u) < 1e-10

        p_at = float(self.probs[mask_at].sum()) if mask_at.any() else 0.0
        alpha_at = float(self.alpha_vals[mask_at][0]) if mask_at.any() else 0.5

        num = (
            float(np.sum(
                self.alpha_vals[mask_above]
                * self.s_vals[mask_above]
                * self.probs[mask_above]
            ))
            + theta_u * alpha_at * q_u * p_at
        )
        denom = (
            float(np.sum(self.s_vals[mask_above] * self.probs[mask_above]))
            + theta_u * q_u * p_at
        )
        if abs(denom) < 1e-300:
            return np.nan
        return num / denom

    def psi_i(self, u: float) -> float:
        """Psi_{1,u} = E[alpha(S)*pi_u(S)] / E[pi_u(S)].

        Since R_1 = X_1/S = alpha_k on atom k, we have
          Psi_{1,u} = sum_{k: s_k > q_u} alpha_k * p_k + theta_u * alpha_{k*} * p_{k*}
                    / sum_{k: s_k > q_u} p_k + theta_u * p_{k*}.
        """
        q_u = self.q_s(u)
        theta_u = self._theta_u(q_u, u)

        mask_above = self.s_vals > q_u
        mask_at = np.abs(self.s_vals - q_u) < 1e-10

        p_at = float(self.probs[mask_at].sum()) if mask_at.any() else 0.0
        alpha_at = float(self.alpha_vals[mask_at][0]) if mask_at.any() else 0.5

        num = (
            float(np.sum(self.alpha_vals[mask_above] * self.probs[mask_above]))
            + theta_u * alpha_at * p_at
        )
        denom = float(np.sum(self.probs[mask_above])) + theta_u * p_at
        if abs(denom) < 1e-300:
            return np.nan
        return num / denom


# ---------------------------------------------------------------------------
# 8. Full diagnostic DataFrame
# ---------------------------------------------------------------------------

def _running_oscillation(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute running oscillation amplitude max(|diff|) over a sliding window.

    At position i the amplitude is the maximum absolute first difference of
    the finite values in values[max(0, i-window+1) : i+1].  Positions with
    fewer than 2 finite values return ``np.nan``.

    Parameters
    ----------
    values:
        1-D array of scalar sequence values (NaN is ignored).
    window:
        Number of trailing elements to include.

    Returns
    -------
    np.ndarray
        Same length as ``values``; entry i is the running oscillation at i.
    """
    n = len(values)
    osc = np.full(n, np.nan)
    for i in range(1, n):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) >= 2:
            osc[i] = float(np.max(np.abs(np.diff(valid))))
    return osc


def compute_asymptotic_diagnostics(
    phi_seq: np.ndarray,
    psi_seq: np.ndarray,
    W_seq: np.ndarray,
    u_seq: np.ndarray,
) -> pd.DataFrame:
    """Assemble a tidy diagnostics DataFrame for one asymptotic scenario.

    Parameters
    ----------
    phi_seq:
        Phi_{i,u} values along the sequence.
    psi_seq:
        Psi_{i,u} values along the sequence.
    W_seq:
        W_u values along the sequence.
    u_seq:
        Probability levels (monotone increasing toward 1).

    Returns
    -------
    pd.DataFrame
        One row per grid point with columns:

        ``u``             – probability level.
        ``phi``           – Phi_{i,u}.
        ``psi``           – Psi_{i,u}.
        ``W``             – W_u.
        ``delta``         – Phi − Psi.
        ``phi_osc``       – running oscillation amplitude of Phi (last 10 steps).
        ``psi_osc``       – running oscillation amplitude of Psi (last 10 steps).
        ``classification`` – overall asymptotic classification string (constant).
    """
    delta = phi_seq - psi_seq
    phi_osc = _running_oscillation(phi_seq)
    psi_osc = _running_oscillation(psi_seq)
    label = classify_asymptotic_behavior(phi_seq, psi_seq, W_seq)

    return pd.DataFrame(
        {
            "u": np.asarray(u_seq, dtype=float),
            "phi": np.asarray(phi_seq, dtype=float),
            "psi": np.asarray(psi_seq, dtype=float),
            "W": np.asarray(W_seq, dtype=float),
            "delta": delta,
            "phi_osc": phi_osc,
            "psi_osc": psi_osc,
            "classification": label,
        }
    )


# ---------------------------------------------------------------------------
# 9. Asymptotic classifier
# ---------------------------------------------------------------------------

def classify_asymptotic_behavior(
    phi_seq: np.ndarray,
    psi_seq: np.ndarray,
    W_seq: np.ndarray,
) -> str:
    """Classify the asymptotic regime of a (Phi, Psi) pair.

    The classification is based on convergence diagnostics (``estimate_convergence``)
    applied to the trailing window of the input sequences.

    Decision logic
    --------------
    1. ``'common_limit'``:
       Both Phi and Psi are converged (oscillation < 1e-4).
    2. ``'phi_converges_psi_oscillates'``:
       Phi is converged but Psi is not.
    3. ``'delta_vanishes_no_limit'``:
       Neither converges individually, but Delta = Phi − Psi converges toward
       zero (i.e., oscillation of Delta < 1e-4 and |mean(Delta)| < 1e-3).
       Also used as the default fallback.

    Parameters
    ----------
    phi_seq:
        Phi_{i,u} values along the sequence.
    psi_seq:
        Psi_{i,u} values along the sequence.
    W_seq:
        W_u values (not used directly in classification, reserved for future
        refinements).

    Returns
    -------
    str
        One of ``'common_limit'``, ``'phi_converges_psi_oscillates'``,
        ``'delta_vanishes_no_limit'``.
    """
    phi_diag = estimate_convergence(phi_seq)
    psi_diag = estimate_convergence(psi_seq)

    if phi_diag["converged"] and psi_diag["converged"]:
        return "common_limit"

    if phi_diag["converged"] and not psi_diag["converged"]:
        return "phi_converges_psi_oscillates"

    # Check whether Delta = Phi - Psi vanishes even though neither converges.
    delta = phi_seq - psi_seq
    delta_diag = estimate_convergence(delta)
    mean_delta = 0.5 * (delta_diag["limsup"] + delta_diag["liminf"])
    if delta_diag["converged"] and abs(mean_delta) < 1e-3:
        return "delta_vanishes_no_limit"

    return "delta_vanishes_no_limit"


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("asymptotic.py — self-test")
    print("=" * 60)

    # ---- Scenario 1: Pareto bounded-W ----
    # Use a log-spaced u-grid very close to 1 so the convergence to the
    # asymptotic limit ell_0 = 1 is visible within the tail window.
    print("\nScenario 1: Pareto bounded-W (alpha=[2,3], sigma=[1,1])")
    scen1 = ParetoBoundedWScenario(alphas=[2.0, 3.0], sigmas=[1.0, 1.0])
    u_seq1 = 1.0 - np.logspace(-3, -10, 60)
    W1 = compute_W_sequence(scen1.q_s, scen1.es_s, u_seq1)
    phi1, psi1 = compute_phi_psi_sequence(
        lambda u: scen1.phi_i(0, u), lambda u: scen1.psi_i(0, u), u_seq1
    )
    diag1 = compute_asymptotic_diagnostics(phi1, psi1, W1, u_seq1)
    print(diag1[["u", "phi", "psi", "W", "delta", "classification"]].tail(5).to_string(index=False))
    print(f"  Asymptotic limit ell_0 = {scen1.asymptotic_limit(0):.6f}  "
          f"(phi tail = {phi1[~np.isnan(phi1)][-1]:.6f})")

    # ---- Scenario 2: Oscillating share ----
    # Dense log-spaced u-grid from 0.9 to 1-1e-8 to resolve many oscillation
    # cycles of sin(2*pi*q_S(u)) while W_u -> 1.
    print("\nScenario 2: OscillatingShare (Exp aggregate, A=0.3, omega=2*pi)")
    scen2 = OscillatingShareScenario(amplitude=0.3, frequency=2.0 * np.pi)
    u_seq2 = 1.0 - np.logspace(-1, -8, 70)
    W2 = compute_W_sequence(scen2.q_s, scen2.es_s, u_seq2)
    phi2, psi2 = compute_phi_psi_sequence(scen2.phi_i, scen2.psi_i, u_seq2)
    diag2 = compute_asymptotic_diagnostics(phi2, psi2, W2, u_seq2)
    print(diag2[["u", "phi", "psi", "W", "delta", "classification"]].tail(5).to_string(index=False))
    print(f"  W -> 1: W_last = {W2[~np.isnan(W2)][-1]:.6f}")
    print(f"  Psi oscillation (last 20): {estimate_convergence(psi2, window=20)['oscillation_amplitude']:.4f}")

    # ---- Scenario 3: Discrete alternating share ----
    # Use N=50 and test at atom indices 5..38 (M = 45..12 atoms above threshold)
    # to stay far from the boundary atom at k=N, where phi is close to 0.5 and
    # psi alternates cleanly between two limits.
    print("\nScenario 3: DiscreteDivergent (N=50, alpha=0.3/0.7)")
    scen3 = DiscreteDivergentWScenario(N=50, alpha_even=0.3, alpha_odd=0.7)
    cumprob3 = np.cumsum(scen3.probs)
    # Mid-atom levels: theta_u = 0.5 at each atom; indices 5..38
    u_seq3 = cumprob3[5:39] - 0.5 * scen3.probs[5:39]
    W3 = compute_W_sequence(scen3.q_s, scen3.es_s, u_seq3)
    phi3, psi3 = compute_phi_psi_sequence(scen3.phi_i, scen3.psi_i, u_seq3)
    diag3 = compute_asymptotic_diagnostics(phi3, psi3, W3, u_seq3)
    print(diag3[["u", "phi", "psi", "W", "delta", "classification"]].tail(6).to_string(index=False))
    print(f"  Psi alternates: {np.round(psi3[-6:], 4)}")
    print(f"  Phi last 6:     {np.round(phi3[-6:], 4)}")

    # ---- Theorem bounds ----
    print("\nTheorem-bound check (Scenario 1):")
    bounds1 = check_theorem_bounds(phi1, psi1, W1)
    n_ok = int(np.sum(bounds1["bound_holds"]))
    print(f"  Bound holds at {n_ok}/{len(phi1)} points; "
          f"max_violation = {bounds1['max_violation']:.2e}")

    print("\nDone.")
