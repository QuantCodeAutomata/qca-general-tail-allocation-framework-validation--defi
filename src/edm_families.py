"""
edm_families.py
===============
Additive Exponential Dispersion Model (EDM) families.

Framework: X_i ~ EDM(lambda_i, theta_i, A) where

    K_{X_i}(t) = lambda_i * [A(theta_i + t) - A(theta_i)]

is the cumulant generating function.  The unit CGF A satisfies:

    A'(theta)  = mean of unit variable
    A''(theta) = variance of unit variable
    iota(theta) = A'(theta) / A''(theta)

Families implemented
--------------------
NormalEDM          – A(theta) = theta^2 / 2
PoissonEDM         – A(theta) = exp(theta)
ScaledPoissonEDM   – A(theta) = (a/b)*exp(b*theta), a*b > 0
GammaEDM           – A(theta) = -log(-theta), theta < 0
TweedieEDM         – power-p family (p=0,1,2,3, 1<p<2 compound Poisson-Gamma)
GenHypSecantEDM    – A(theta) = -log(cos(theta)), |theta| < pi/2

Module-level utility functions
-------------------------------
compute_saddle_point       – solve K_S'(t_hat) = s
compute_g_hat_i            – saddle-point conditional share proxy
compute_cumulant_aggregate – K_S(t)
compute_aggregate_mean     – E[S]
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize, special
from typing import Optional, Tuple, Dict, Callable
import warnings


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class EDMFamily:
    """Abstract base for additive EDM families.

    Every concrete family must implement A, A_prime, A_double_prime,
    theta_domain, sample_aggregate, and sample_components.  The default
    iota uses the ratio A'(theta) / A''(theta).
    """

    def A(self, theta: float) -> float:
        """Unit cumulant generating function A(theta)."""
        raise NotImplementedError

    def A_prime(self, theta: float) -> float:
        """First derivative A'(theta) = E[unit variable]."""
        raise NotImplementedError

    def A_double_prime(self, theta: float) -> float:
        """Second derivative A''(theta) = Var[unit variable]."""
        raise NotImplementedError

    def iota(self, theta: float) -> float:
        """Canonical ratio iota(theta) = A'(theta) / A''(theta)."""
        return self.A_prime(theta) / self.A_double_prime(theta)

    def theta_domain(self) -> Tuple[float, float]:
        """Return open interval (lo, hi) that is the natural parameter domain."""
        raise NotImplementedError

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S = sum_i X_i.

        Parameters
        ----------
        lambdas, thetas:
            EDM parameters; shape (n,).
        n_samples:
            Number of i.i.d. realisations of S to draw.
        rng:
            NumPy random Generator (e.g. np.random.default_rng(42)).

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        raise NotImplementedError

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample each X_i independently.

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
            Row i contains n_samples draws from EDM(lambda_i, theta_i, A).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Normal EDM
# ---------------------------------------------------------------------------

class NormalEDM(EDMFamily):
    """Normal EDM family.

    Unit CGF: A(theta) = theta^2 / 2, theta in R.

    Under the reparameterization mu_i = sigma_i^2 * theta_i,
    lambda_i = sigma_i^2:

        X_i ~ N(mu_i, sigma_i^2)

    and the aggregate is:

        S = sum_i X_i ~ N(mu_S, sigma_S^2)

    with mu_S = sum mu_i, sigma_S^2 = sum sigma_i^2.
    """

    def A(self, theta: float) -> float:
        """A(theta) = theta^2 / 2."""
        return 0.5 * theta ** 2

    def A_prime(self, theta: float) -> float:
        """A'(theta) = theta."""
        return float(theta)

    def A_double_prime(self, theta: float) -> float:
        """A''(theta) = 1."""
        return 1.0

    def iota(self, theta: float) -> float:
        """iota(theta) = theta (identity map)."""
        return float(theta)

    def theta_domain(self) -> Tuple[float, float]:
        """Full real line (-inf, +inf)."""
        return (-np.inf, np.inf)

    # ------------------------------------------------------------------
    # Distribution-specific methods
    # ------------------------------------------------------------------

    def exact_g_i(
        self,
        s: float,
        i: int,
        mus: np.ndarray,
        sigmas: np.ndarray,
    ) -> float:
        """Exact conditional share g_i(s) = E[X_i | S = s] / s.

        Under independence and normality:

            E[X_i | S = s] = mu_i + beta_i * (s - mu_S)
            beta_i = sigma_i^2 / sigma_S^2

        So:

            g_i(s) = beta_i + (mu_i - beta_i * mu_S) / s   (s != 0)

        Parameters
        ----------
        s : float
            Realised aggregate value (must be non-zero).
        i : int
            Component index.
        mus, sigmas : np.ndarray
            Component means and standard deviations.

        Returns
        -------
        float
            g_i(s) in [0, 1] (may exceed these bounds in the tail).
        """
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        mu_s = float(np.sum(mus))
        sigma_s_sq = float(np.sum(sigmas ** 2))
        if sigma_s_sq == 0.0:
            raise ValueError("sigma_S^2 = 0; all components are degenerate.")
        beta_i = sigmas[i] ** 2 / sigma_s_sq
        if s == 0.0:
            warnings.warn("s = 0: g_i undefined; returning nan.", RuntimeWarning, stacklevel=2)
            return np.nan
        return beta_i + (mus[i] - beta_i * mu_s) / s

    def aggregate_cdf(
        self,
        s: float,
        mus: np.ndarray,
        sigmas: np.ndarray,
    ) -> float:
        """CDF of aggregate S ~ N(mu_S, sigma_S^2).

        Parameters
        ----------
        s : float
            Evaluation point.
        mus, sigmas : np.ndarray
            Component means and standard deviations.

        Returns
        -------
        float
            P(S <= s).
        """
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        mu_s = float(np.sum(mus))
        sigma_s = float(np.sqrt(np.sum(sigmas ** 2)))
        return float(stats.norm.cdf(s, loc=mu_s, scale=sigma_s))

    def aggregate_ppf(
        self,
        u: float,
        mus: np.ndarray,
        sigmas: np.ndarray,
    ) -> float:
        """Quantile function (PPF) of aggregate S ~ N(mu_S, sigma_S^2).

        Parameters
        ----------
        u : float
            Probability level in (0, 1).
        mus, sigmas : np.ndarray
            Component means and standard deviations.

        Returns
        -------
        float
            q_S(u) = F_S^{-1}(u).
        """
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        mu_s = float(np.sum(mus))
        sigma_s = float(np.sqrt(np.sum(sigmas ** 2)))
        return float(stats.norm.ppf(u, loc=mu_s, scale=sigma_s))

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample X_i ~ N(lambda_i * theta_i, lambda_i) independently.

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        n = len(lambdas)
        out = np.empty((n, n_samples), dtype=float)
        for i in range(n):
            mu_i = lambdas[i] * thetas[i]
            std_i = np.sqrt(lambdas[i])
            out[i] = rng.normal(loc=mu_i, scale=std_i, size=n_samples)
        return out

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S ~ N(mu_S, sigma_S^2) directly.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        mu_s = float(np.sum(lambdas * thetas))
        sigma_s = float(np.sqrt(np.sum(lambdas)))
        return rng.normal(loc=mu_s, scale=sigma_s, size=n_samples)


# ---------------------------------------------------------------------------
# 2. Poisson EDM
# ---------------------------------------------------------------------------

class PoissonEDM(EDMFamily):
    """Poisson EDM family (scaled Poisson with a = b = 1).

    Unit CGF: A(theta) = exp(theta), theta in R.

        X_i ~ Poisson(lambda_i * exp(theta_i))

    The aggregate of independent Poissons is:

        S ~ Poisson(mu_S),  mu_S = sum_i lambda_i * exp(theta_i)

    Equality weight
    ---------------
    Homogeneous case (all theta_i equal): r_i = lambda_i / sum lambda_j.
    Heterogeneous case:  r_i = lambda_i * exp(theta_i) / sum lambda_j * exp(theta_j).
    """

    def A(self, theta: float) -> float:
        """A(theta) = exp(theta)."""
        return np.exp(theta)

    def A_prime(self, theta: float) -> float:
        """A'(theta) = exp(theta)."""
        return np.exp(theta)

    def A_double_prime(self, theta: float) -> float:
        """A''(theta) = exp(theta)."""
        return np.exp(theta)

    def iota(self, theta: float) -> float:
        """iota(theta) = 1 (constant)."""
        return 1.0

    def theta_domain(self) -> Tuple[float, float]:
        """Full real line (-inf, +inf)."""
        return (-np.inf, np.inf)

    # ------------------------------------------------------------------
    # Distribution-specific methods
    # ------------------------------------------------------------------

    def _aggregate_rate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
    ) -> float:
        """mu_S = sum_i lambda_i * exp(theta_i)."""
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        return float(np.sum(lambdas * np.exp(thetas)))

    def aggregate_cdf(
        self,
        s: float,
        lambdas: np.ndarray,
        thetas: np.ndarray,
    ) -> float:
        """CDF of aggregate S ~ Poisson(mu_S).

        Parameters
        ----------
        s : float
            Evaluation point (treated as floor(s) for integer support).
        lambdas, thetas : np.ndarray
            EDM parameters.

        Returns
        -------
        float
            P(S <= s).
        """
        mu = self._aggregate_rate(lambdas, thetas)
        return float(stats.poisson.cdf(s, mu=mu))

    def aggregate_pmf(
        self,
        s: float,
        lambdas: np.ndarray,
        thetas: np.ndarray,
    ) -> float:
        """PMF of aggregate S ~ Poisson(mu_S).

        Parameters
        ----------
        s : float
            Non-negative integer evaluation point.
        lambdas, thetas : np.ndarray
            EDM parameters.

        Returns
        -------
        float
            P(S = s).
        """
        mu = self._aggregate_rate(lambdas, thetas)
        return float(stats.poisson.pmf(s, mu=mu))

    def equality_weights_homogeneous(self, lambdas: np.ndarray) -> np.ndarray:
        """Equality weights r_i = lambda_i / sum lambda_j when all theta_i equal.

        Parameters
        ----------
        lambdas : np.ndarray
            EDM dispersion parameters.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        return lambdas / lambdas.sum()

    def equality_weights_heterogeneous(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
    ) -> np.ndarray:
        """Equality weights r_i = lambda_i*exp(theta_i) / sum lambda_j*exp(theta_j).

        Parameters
        ----------
        lambdas, thetas : np.ndarray
            EDM parameters.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        rates = lambdas * np.exp(thetas)
        return rates / rates.sum()

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample X_i ~ Poisson(lambda_i * exp(theta_i)) independently.

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        n = len(lambdas)
        out = np.empty((n, n_samples), dtype=float)
        for i in range(n):
            rate_i = lambdas[i] * np.exp(thetas[i])
            out[i] = rng.poisson(lam=rate_i, size=n_samples).astype(float)
        return out

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S ~ Poisson(mu_S) directly.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        mu = self._aggregate_rate(lambdas, thetas)
        return rng.poisson(lam=mu, size=n_samples).astype(float)


# ---------------------------------------------------------------------------
# 3. Scaled Poisson EDM
# ---------------------------------------------------------------------------

class ScaledPoissonEDM(EDMFamily):
    """Scaled Poisson EDM family with shape parameters (a, b), a*b > 0.

    Unit CGF: A(theta) = (a/b) * exp(b * theta).

        A'(theta)  = a * exp(b * theta)
        A''(theta) = a * b * exp(b * theta)
        iota(theta) = 1 / b  (constant)

    For a = b = 1 this reduces to the standard PoissonEDM.

    The condition a*b > 0 ensures A is convex (A'' > 0).

    Parameters
    ----------
    a, b : float
        Shape parameters satisfying a * b > 0.
    """

    def __init__(self, a: float, b: float) -> None:
        if a * b <= 0:
            raise ValueError(f"Require a*b > 0; got a={a}, b={b}.")
        self.a = float(a)
        self.b = float(b)

    def A(self, theta: float) -> float:
        """A(theta) = (a/b) * exp(b*theta)."""
        return (self.a / self.b) * np.exp(self.b * theta)

    def A_prime(self, theta: float) -> float:
        """A'(theta) = a * exp(b*theta)."""
        return self.a * np.exp(self.b * theta)

    def A_double_prime(self, theta: float) -> float:
        """A''(theta) = a*b * exp(b*theta)."""
        return self.a * self.b * np.exp(self.b * theta)

    def iota(self, theta: float) -> float:
        """iota(theta) = 1/b (constant)."""
        return 1.0 / self.b

    def theta_domain(self) -> Tuple[float, float]:
        """Full real line (-inf, +inf)."""
        return (-np.inf, np.inf)

    def equality_weights(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
    ) -> np.ndarray:
        """Global equality weights.

        r_i = lambda_i * exp(b * theta_i) / sum_j lambda_j * exp(b * theta_j)

        These are the proportional shares of the aggregate mean.

        Parameters
        ----------
        lambdas, thetas : np.ndarray
            EDM parameters.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        weights = lambdas * np.exp(self.b * thetas)
        return weights / weights.sum()

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample each component via a compound Poisson representation.

        Since K_{X_i}(t) = lambda_i * (a/b) * [exp(b*(theta_i+t)) - exp(b*theta_i)],
        X_i ~ ScaledPoisson(lambda_i, theta_i, a, b) is sampled by noting that
        b*X_i ~ Poisson-family.  We use normal approximation for large rates
        and exact Poisson simulation for small rates.

        Concretely: let Y_i = b * X_i, then
            K_{Y_i}(t) = K_{X_i}(b*t) = lambda_i*(a/b)*[exp(b*(theta_i+b*t))-exp(b*theta_i)]

        For exact sampling we rely on the unit-variable decomposition:
        the unit variable with parameter (a, b) has the same shape as a scaled
        Poisson with rate a/b and scale 1/b, so we sample via:

            X_i = (1/b) * Poisson(lambda_i * a * exp(b * theta_i))

        when b > 0.  For b < 0 (and a < 0 so a*b > 0) the argument of the
        exponential reverses roles and we still have a valid Poisson rate
        lambda_i * a * exp(b * theta_i) > 0.

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        n = len(lambdas)
        out = np.empty((n, n_samples), dtype=float)
        for i in range(n):
            # Poisson rate for the scaled representation
            rate_i = lambdas[i] * self.a * np.exp(self.b * thetas[i])
            if rate_i <= 0:
                raise ValueError(
                    f"Poisson rate for component {i} is non-positive ({rate_i}). "
                    "Check that a*b > 0 and parameter values are valid."
                )
            poisson_draw = rng.poisson(lam=rate_i, size=n_samples).astype(float)
            out[i] = poisson_draw / self.b
        return out

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S = sum_i X_i by summing independent component samples.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        components = self.sample_components(lambdas, thetas, n_samples, rng)
        return components.sum(axis=0)


# ---------------------------------------------------------------------------
# 4. Gamma EDM
# ---------------------------------------------------------------------------

class GammaEDM(EDMFamily):
    """Gamma EDM family.

    Unit CGF: A(theta) = -log(-theta), theta < 0.

        A'(theta)  = -1 / theta   (unit mean)
        A''(theta) = 1 / theta^2  (unit variance)
        iota(theta) = A'(theta)/A''(theta) = -theta

    Under the EDM parameterization:

        X_i ~ Gamma(shape=lambda_i, rate=-theta_i)

    so E[X_i] = lambda_i / (-theta_i) and Var[X_i] = lambda_i / theta_i^2.

    Homogeneous case: all theta_i = theta.

        S ~ Gamma(shape=sum lambda_i, rate=-theta)

    with g_i^hom(s) = lambda_i / sum lambda_j (constant in s).
    """

    def A(self, theta: float) -> float:
        """A(theta) = -log(-theta), theta < 0."""
        if theta >= 0:
            raise ValueError(f"Gamma EDM requires theta < 0; got theta={theta}.")
        return -np.log(-theta)

    def A_prime(self, theta: float) -> float:
        """A'(theta) = -1/theta."""
        if theta >= 0:
            raise ValueError(f"Gamma EDM requires theta < 0; got theta={theta}.")
        return -1.0 / theta

    def A_double_prime(self, theta: float) -> float:
        """A''(theta) = 1/theta^2."""
        if theta >= 0:
            raise ValueError(f"Gamma EDM requires theta < 0; got theta={theta}.")
        return 1.0 / theta ** 2

    def iota(self, theta: float) -> float:
        """iota(theta) = -theta (positive since theta < 0)."""
        if theta >= 0:
            raise ValueError(f"Gamma EDM requires theta < 0; got theta={theta}.")
        return -theta

    def theta_domain(self) -> Tuple[float, float]:
        """Domain is (-inf, 0) (open at both ends)."""
        return (-np.inf, 0.0)

    # ------------------------------------------------------------------
    # Distribution-specific methods
    # ------------------------------------------------------------------

    def g_i_exact_homogeneous(
        self,
        s: float,
        i: int,
        lambdas: np.ndarray,
        theta: float,
    ) -> float:
        """Exact conditional share in the homogeneous case (all theta_i = theta).

        When all theta_i = theta, S ~ Gamma(sum lambda_j, rate=-theta), and
        E[X_i | S = s] = s * lambda_i / sum lambda_j, so:

            g_i^hom(s) = lambda_i / sum lambda_j  (constant in s)

        Parameters
        ----------
        s : float
            Realised aggregate (unused in the constant formula, kept for API).
        i : int
            Component index.
        lambdas : np.ndarray
            Shape parameters.
        theta : float
            Common natural parameter (< 0).

        Returns
        -------
        float
        """
        if theta >= 0:
            raise ValueError(f"Gamma EDM requires theta < 0; got theta={theta}.")
        lambdas = np.asarray(lambdas, dtype=float)
        return float(lambdas[i] / lambdas.sum())

    def aggregate_cdf_n2(
        self,
        s: float,
        lambda1: float,
        lambda2: float,
        theta1: float,
        theta2: float,
    ) -> float:
        """CDF of S = X_1 + X_2 for two independent Gamma components.

        Homogeneous case (theta1 = theta2 = theta):
            S ~ Gamma(lambda1 + lambda2, rate = -theta); exact formula used.

        Heterogeneous case (theta1 != theta2):
            S has a generalised Gamma (hypoexponential) distribution.
            The PDF is computed via numerical integration and the CDF
            is obtained by integrating the PDF from 0 to s.

        Parameters
        ----------
        s : float
            Evaluation point (s > 0).
        lambda1, lambda2 : float
            Shape parameters (> 0).
        theta1, theta2 : float
            Rate parameters (< 0; rate = -theta).

        Returns
        -------
        float
            P(S <= s).
        """
        if theta1 >= 0 or theta2 >= 0:
            raise ValueError("Gamma EDM requires theta1 < 0 and theta2 < 0.")
        if s <= 0.0:
            return 0.0

        rate1 = -theta1
        rate2 = -theta2

        if np.isclose(theta1, theta2, rtol=1e-10, atol=1e-14):
            # Exact: S ~ Gamma(lambda1+lambda2, rate=rate1)
            return float(stats.gamma.cdf(s, a=lambda1 + lambda2, scale=1.0 / rate1))

        # Heterogeneous: convolve via numerical integration.
        # f_S(s) = integral_0^s f_{X1}(x) * f_{X2}(s-x) dx
        # Integrate and accumulate using scipy.
        def pdf_s(t: float) -> float:
            if t <= 0.0 or t >= s:
                return 0.0
            f1 = stats.gamma.pdf(t, a=lambda1, scale=1.0 / rate1)
            f2 = stats.gamma.pdf(s - t, a=lambda2, scale=1.0 / rate2)
            return f1 * f2

        # CDF = integral_0^s f_S(t) dt
        cdf_val, _ = integrate.quad(
            lambda t: stats.gamma.pdf(t, a=lambda1, scale=1.0 / rate1)
                      * stats.gamma.pdf(s - t, a=lambda2, scale=1.0 / rate2),
            0.0,
            s,
            limit=200,
        )
        return float(np.clip(cdf_val, 0.0, 1.0))

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample X_i ~ Gamma(shape=lambda_i, rate=-theta_i).

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        if np.any(thetas >= 0):
            raise ValueError("Gamma EDM requires all theta_i < 0.")
        n = len(lambdas)
        out = np.empty((n, n_samples), dtype=float)
        for i in range(n):
            rate_i = -thetas[i]
            # Gamma(shape=lambda_i, scale=1/rate_i)
            out[i] = rng.gamma(shape=lambdas[i], scale=1.0 / rate_i, size=n_samples)
        return out

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S = sum_i X_i.

        Homogeneous case uses the exact Gamma(sum lambda_i, -theta).
        Heterogeneous case sums independent Gamma samples.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        if np.any(thetas >= 0):
            raise ValueError("Gamma EDM requires all theta_i < 0.")

        if np.allclose(thetas, thetas[0]):
            rate = -thetas[0]
            return rng.gamma(shape=float(lambdas.sum()), scale=1.0 / rate, size=n_samples)

        # Heterogeneous: sum independent draws
        components = self.sample_components(lambdas, thetas, n_samples, rng)
        return components.sum(axis=0)


# ---------------------------------------------------------------------------
# 5. Tweedie EDM
# ---------------------------------------------------------------------------

class TweedieEDM(EDMFamily):
    """Tweedie EDM family parameterised by power p.

    Supported values of p
    ---------------------
    p = 0   Normal          A(theta) = theta^2/2,     theta in R
    p = 1   Poisson         A(theta) = exp(theta),    theta in R
    p = 2   Gamma           A(theta) = -log(-theta),  theta < 0
    p = 3   Inv. Gaussian   A(theta) = -sqrt(-2*theta), theta < 0
    1<p<2   Compound        A(theta) = [(1-p)*theta]^{kappa}/(2-p)
            Poisson-Gamma    with kappa=(2-p)/(1-p), theta < 0

    General formula (p != 0, 1, 2)
    --------------------------------
    Let kappa = (2 - p) / (1 - p).

    For 1 < p < 2 and p > 2: theta < 0, and (1-p)*theta > 0.

        A(theta)   = [(1-p)*theta]^kappa / (2-p)
        A'(theta)  = [(1-p)*theta]^{kappa-1}
        A''(theta) = (kappa-1)*(1-p)*[(1-p)*theta]^{kappa-2}
                   = [(1-p)*theta]^{1/(1-p)-1}   (simplified)
        iota(theta) = (1-p)*theta

    For p < 0 or 0 < p < 1: theta > 0, and (1-p)*theta > 0.
    (Here these sub-cases are not explicitly exposed but the math is consistent.)

    Parameters
    ----------
    p : float
        Tweedie power.  Supported: 0, 1, 2, 3, or any float in (1, 2).
    """

    _SUPPORTED_P = frozenset({0.0, 1.0, 2.0, 3.0})

    def __init__(self, p: float) -> None:
        self.p = float(p)
        if not (
            np.isclose(self.p, 0.0)
            or np.isclose(self.p, 1.0)
            or np.isclose(self.p, 2.0)
            or np.isclose(self.p, 3.0)
            or (1.0 < self.p < 2.0)
        ):
            raise ValueError(
                f"TweedieEDM supports p in {{0,1,2,3}} or 1<p<2; got p={self.p}."
            )
        # Delegate to sub-family helpers where convenient
        if np.isclose(self.p, 0.0):
            self._normal = NormalEDM()
        elif np.isclose(self.p, 1.0):
            self._poisson = PoissonEDM()
        elif np.isclose(self.p, 2.0):
            self._gamma = GammaEDM()

    # ------------------------------------------------------------------
    # Core EDM interface
    # ------------------------------------------------------------------

    def A(self, theta: float) -> float:
        """Unit CGF A(theta) for the specified power p."""
        p = self.p
        if np.isclose(p, 0.0):
            return 0.5 * theta ** 2
        if np.isclose(p, 1.0):
            return np.exp(theta)
        if np.isclose(p, 2.0):
            if theta >= 0:
                raise ValueError("Gamma (p=2) requires theta < 0.")
            return -np.log(-theta)
        # General case: p=3 and 1<p<2 and p>2
        if theta >= 0 and not np.isclose(p, 0.0):
            raise ValueError(f"Tweedie p={p} requires theta < 0.")
        base = (1.0 - p) * theta    # > 0 when theta < 0 and p > 1
        if base <= 0:
            raise ValueError(
                f"(1-p)*theta = {base} <= 0; invalid for Tweedie p={p}, theta={theta}."
            )
        kappa = (2.0 - p) / (1.0 - p)
        return base ** kappa / (2.0 - p)

    def A_prime(self, theta: float) -> float:
        """A'(theta) = unit mean."""
        p = self.p
        if np.isclose(p, 0.0):
            return float(theta)
        if np.isclose(p, 1.0):
            return np.exp(theta)
        if np.isclose(p, 2.0):
            if theta >= 0:
                raise ValueError("Gamma (p=2) requires theta < 0.")
            return -1.0 / theta
        # General: A'(theta) = [(1-p)*theta]^{kappa-1}
        if theta >= 0:
            raise ValueError(f"Tweedie p={p} requires theta < 0.")
        base = (1.0 - p) * theta
        kappa = (2.0 - p) / (1.0 - p)
        return base ** (kappa - 1.0)

    def A_double_prime(self, theta: float) -> float:
        """A''(theta) = unit variance."""
        p = self.p
        if np.isclose(p, 0.0):
            return 1.0
        if np.isclose(p, 1.0):
            return np.exp(theta)
        if np.isclose(p, 2.0):
            if theta >= 0:
                raise ValueError("Gamma (p=2) requires theta < 0.")
            return 1.0 / theta ** 2
        # General: A''(theta) = (kappa-1)*(1-p)*base^{kappa-2}
        # Equivalently = base^{1/(1-p)-1} using kappa-1 = 1/(1-p)
        if theta >= 0:
            raise ValueError(f"Tweedie p={p} requires theta < 0.")
        base = (1.0 - p) * theta
        kappa = (2.0 - p) / (1.0 - p)
        return (kappa - 1.0) * (1.0 - p) * base ** (kappa - 2.0)

    def iota(self, theta: float) -> float:
        """iota(theta) = A'(theta) / A''(theta) = (1-p)*theta (general p).

        Special cases:
            p=0: iota = theta
            p=1: iota = 1
            p=2: iota = -theta
        """
        p = self.p
        if np.isclose(p, 0.0):
            return float(theta)
        if np.isclose(p, 1.0):
            return 1.0
        if np.isclose(p, 2.0):
            if theta >= 0:
                raise ValueError("Gamma (p=2) requires theta < 0.")
            return -theta
        # General: iota = (1-p)*theta
        if theta >= 0:
            raise ValueError(f"Tweedie p={p} requires theta < 0.")
        return (1.0 - p) * theta

    def theta_domain(self) -> Tuple[float, float]:
        """Natural parameter domain.

        p=0: (-inf, +inf)
        p=1: (-inf, +inf)
        p>=2 or 1<p<2: (-inf, 0)
        """
        p = self.p
        if np.isclose(p, 0.0) or np.isclose(p, 1.0):
            return (-np.inf, np.inf)
        return (-np.inf, 0.0)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_compound_poisson_gamma(
        self,
        lambda_i: float,
        theta_i: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one Tweedie component for 1 < p < 2 (compound Poisson-Gamma).

        Compound representation of the unit variable Z (lambda=1):
            N ~ Poisson(alpha),   alpha = A(theta_i) = [(1-p)*theta]^kappa/(2-p)
            W_j ~ Gamma(r, rate=beta),  r = (2-p)/(p-1), beta = -theta_i
            Z = sum_{j=1}^N W_j

        For X_i with dispersion lambda_i, the count is Poisson(lambda_i * alpha)
        with the same Gamma jump distribution.

        Parameters
        ----------
        lambda_i : float
            Dispersion parameter for component i.
        theta_i : float
            Natural parameter (< 0).
        n_samples : int

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        p = self.p
        kappa = (2.0 - p) / (1.0 - p)   # < 0 for 1 < p < 2
        base = (1.0 - p) * theta_i       # > 0

        # Unit Poisson rate (= A(theta_i))
        alpha_unit = base ** kappa / (2.0 - p)

        # Gamma jump parameters: shape r, rate beta
        r = (2.0 - p) / (p - 1.0)        # > 0
        beta = -theta_i                   # > 0

        # Scaled Poisson rate
        poisson_rate = lambda_i * alpha_unit

        # Draw counts
        counts = rng.poisson(lam=poisson_rate, size=n_samples)
        total_count = int(counts.sum())

        # Draw all jumps at once, then assign
        if total_count == 0:
            return np.zeros(n_samples, dtype=float)

        all_jumps = rng.gamma(shape=r, scale=1.0 / beta, size=total_count)

        out = np.zeros(n_samples, dtype=float)
        idx = 0
        for k in range(n_samples):
            c = counts[k]
            if c > 0:
                out[k] = all_jumps[idx: idx + c].sum()
                idx += c
        return out

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample each component according to the Tweedie family for power p.

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        n = len(lambdas)
        out = np.empty((n, n_samples), dtype=float)
        p = self.p

        for i in range(n):
            lam, th = lambdas[i], thetas[i]
            if np.isclose(p, 0.0):
                # X_i ~ N(lam*th, lam)
                out[i] = rng.normal(loc=lam * th, scale=np.sqrt(lam), size=n_samples)
            elif np.isclose(p, 1.0):
                # X_i ~ Poisson(lam * exp(th))
                out[i] = rng.poisson(lam=lam * np.exp(th), size=n_samples).astype(float)
            elif np.isclose(p, 2.0):
                # X_i ~ Gamma(shape=lam, rate=-th)
                out[i] = rng.gamma(shape=lam, scale=1.0 / (-th), size=n_samples)
            elif np.isclose(p, 3.0):
                # Inverse Gaussian: E[X]=lam/(-th), shape=lam^2*(-th)^{???}
                # Standard form: InvGaussian(mu_ig, lambda_ig)
                # E[X_i] = lam * A'(th) = lam * 1/sqrt(-2*th)
                # Var[X_i] = lam * A''(th) = lam * (-2*th)^{-3/2}
                # Scipy invgauss: X ~ invgauss(mu, loc=0, scale=lambda_ig)
                #   where E = mu*lambda_ig, Var = mu^3 * lambda_ig
                mu_x = lam / np.sqrt(-2.0 * th)      # lam * A'(th)
                var_x = lam * (-2.0 * th) ** (-1.5)  # lam * A''(th)
                # Parameterise: mu_ig = E/var^{1/3}... Actually use:
                # scipy invgauss: pdf proportional to exp(-(x-mu_ig)^2/(2*mu_ig^2*x))
                # E = mu_ig * scale, Var = mu_ig^3 * scale
                # => scale = var_x / mu_x^3 * ... let's solve properly:
                # E = mu_ig_scipy * scale, Var = mu_ig_scipy^3 * scale
                # mu_ig_scipy = E/scale, so Var = (E/scale)^3 * scale = E^3/scale^2
                # => scale = sqrt(E^3 / Var)
                scale_ig = np.sqrt(mu_x ** 3 / var_x)
                mu_ig = mu_x / scale_ig
                out[i] = stats.invgauss.rvs(
                    mu=mu_ig, scale=scale_ig, size=n_samples, random_state=rng
                )
            elif 1.0 < p < 2.0:
                out[i] = self._sample_compound_poisson_gamma(lam, th, n_samples, rng)
            else:
                raise NotImplementedError(f"Sampling not implemented for p={p}.")
        return out

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S = sum_i X_i by summing independent component samples.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        return self.sample_components(lambdas, thetas, n_samples, rng).sum(axis=0)


# ---------------------------------------------------------------------------
# 6. Generalised Hyperbolic Secant EDM
# ---------------------------------------------------------------------------

class GenHypSecantEDM(EDMFamily):
    """Generalised Hyperbolic Secant EDM family.

    Unit CGF: A(theta) = -log(cos(theta)) = log(sec(theta)), |theta| < pi/2.

        A'(theta)  = tan(theta)
        A''(theta) = sec^2(theta) = 1/cos^2(theta)
        iota(theta) = sin(theta)*cos(theta) = sin(2*theta) / 2

    The unit variable has the hyperbolic secant distribution.

    Switching-tilt strategy
    -----------------------
    The switching tilt t* shifts the natural parameter to pi/4 via

        t* = pi/4 - theta_min,   theta_min = min_i theta_i

    which moves the "lightest-tail" component to the canonical pivot.
    The corresponding aggregate threshold u* = K_S'(t*) locates the
    level below which the tilt is active.

    Parameters
    ----------
    (none; the domain is fixed at |theta| < pi/2)
    """

    _HALF_PI = 0.5 * np.pi

    def A(self, theta: float) -> float:
        """A(theta) = -log(cos(theta)), |theta| < pi/2."""
        if abs(theta) >= self._HALF_PI:
            raise ValueError(f"|theta| must be < pi/2; got theta={theta}.")
        return -np.log(np.cos(theta))

    def A_prime(self, theta: float) -> float:
        """A'(theta) = tan(theta)."""
        if abs(theta) >= self._HALF_PI:
            raise ValueError(f"|theta| must be < pi/2; got theta={theta}.")
        return np.tan(theta)

    def A_double_prime(self, theta: float) -> float:
        """A''(theta) = sec^2(theta) = 1/cos^2(theta)."""
        if abs(theta) >= self._HALF_PI:
            raise ValueError(f"|theta| must be < pi/2; got theta={theta}.")
        cos_th = np.cos(theta)
        return 1.0 / (cos_th ** 2)

    def iota(self, theta: float) -> float:
        """iota(theta) = sin(theta)*cos(theta) = sin(2*theta)/2."""
        if abs(theta) >= self._HALF_PI:
            raise ValueError(f"|theta| must be < pi/2; got theta={theta}.")
        return np.sin(theta) * np.cos(theta)

    def theta_domain(self) -> Tuple[float, float]:
        """Open interval (-pi/2, pi/2)."""
        return (-self._HALF_PI, self._HALF_PI)

    # ------------------------------------------------------------------
    # Switching-tilt utilities
    # ------------------------------------------------------------------

    def switching_tilt(self, thetas: np.ndarray) -> float:
        """Compute the switching tilt t* = pi/4 - min_i theta_i.

        The tilt t* is the value that shifts the minimum theta_i to pi/4,
        placing the system at the canonical hyperbolic secant pivot.

        Parameters
        ----------
        thetas : np.ndarray
            Natural parameters for all components.

        Returns
        -------
        float
            t* = pi/4 - min(thetas).
        """
        thetas = np.asarray(thetas, dtype=float)
        theta_min = float(np.min(thetas))
        return np.pi / 4.0 - theta_min

    def switching_threshold(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
    ) -> float:
        """Compute the aggregate threshold u* = K_S'(t*).

        At the switching tilt t* the derivative of K_S is:

            u* = sum_i lambda_i * tan(theta_i + t*)

        This is the aggregate level below which the localisation tilt t*
        is applied.

        Parameters
        ----------
        lambdas, thetas : np.ndarray
            EDM parameters.

        Returns
        -------
        float
            u* = E_S evaluated at the switching tilt.
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        t_star = self.switching_tilt(thetas)
        tilted_thetas = thetas + t_star
        if np.any(np.abs(tilted_thetas) >= self._HALF_PI):
            raise ValueError(
                "Tilted theta_i + t* falls outside (-pi/2, pi/2). "
                "Reduce the magnitude of thetas."
            )
        return float(np.sum(lambdas * np.tan(tilted_thetas)))

    def sample_unit(self, theta: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample the unit hyperbolic secant variable at parameter theta.

        The standard hyperbolic secant distribution has PDF
            f(x) = (1/pi) * sech(x),  x in R,
        and characteristic function cos(t)^{-1} = sec(t) for |t| < pi/2,
        matching our A(theta).

        For theta != 0 we use a tilted version: if Z ~ HypSec(0), then
        the tilted density at theta is f_theta(x) ∝ exp(theta*x)*f(x).

        Sampling is performed by inversion of the hyperbolic secant CDF.
        The CDF of the standard distribution is:
            F(x) = (2/pi)*arctan(exp(x))

        so x = log(tan(pi*u/2)).

        For the tilted version we use rejection sampling with an envelope
        from a logistic distribution (which has the same tail).

        Parameters
        ----------
        theta : float
            Natural parameter; |theta| < pi/2.
        n_samples : int

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if abs(theta) >= self._HALF_PI:
            raise ValueError(f"|theta| must be < pi/2; got theta={theta}.")

        # Standard HypSec: F(x) = (2/pi)*arctan(exp(x)) => quantile = log(tan(pi*u/2))
        u = rng.uniform(0.0, 1.0, size=n_samples)
        # Avoid boundary issues
        u = np.clip(u, 1e-15, 1.0 - 1e-15)
        z_std = np.log(np.tan(0.5 * np.pi * u))   # standard HypSec samples

        if np.isclose(theta, 0.0):
            return z_std

        # Importance-sampling correction for theta != 0:
        # X_theta ~ f_theta(x) = exp(theta*x - A(theta)) * f(x)
        # We have z_std ~ f(x).  Weight: exp(theta*z - A(theta)).
        # Use SIR (sampling-importance resampling).
        log_weights = theta * z_std - self.A(theta)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()

        indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights)
        return z_std[indices]

    def sample_components(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample X_i = sum_{j=1}^{lambda_i} Z_{i,j} where Z_{i,j} are unit HypSec.

        For integer lambda_i this is exact.  For non-integer lambda_i we
        approximate by sampling lambda_i unit variables and summing.

        Returns
        -------
        np.ndarray, shape (len(lambdas), n_samples)
        """
        lambdas = np.asarray(lambdas, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        n = len(lambdas)
        out = np.empty((n, n_samples), dtype=float)

        for i in range(n):
            lam = lambdas[i]
            th = thetas[i]
            lam_int = int(round(lam))
            if lam_int < 1:
                lam_int = 1
            # Sum lam_int unit HypSec(theta) variables
            acc = np.zeros(n_samples, dtype=float)
            for _ in range(lam_int):
                acc += self.sample_unit(th, n_samples, rng)
            # Scale if lam != lam_int
            if not np.isclose(lam, float(lam_int)):
                acc *= lam / float(lam_int)
            out[i] = acc
        return out

    def sample_aggregate(
        self,
        lambdas: np.ndarray,
        thetas: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample S = sum_i X_i.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        return self.sample_components(lambdas, thetas, n_samples, rng).sum(axis=0)


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def compute_cumulant_aggregate(
    family: EDMFamily,
    t: float,
    lambdas: np.ndarray,
    thetas: np.ndarray,
) -> float:
    """Cumulant generating function of the aggregate S evaluated at t.

    K_S(t) = sum_i lambda_i * [A(theta_i + t) - A(theta_i)]

    Parameters
    ----------
    family : EDMFamily
        An instantiated EDM family.
    t : float
        Evaluation point (tilt).
    lambdas, thetas : np.ndarray
        EDM parameters; shape (n,).

    Returns
    -------
    float
        K_S(t).
    """
    lambdas = np.asarray(lambdas, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    total = 0.0
    for lam, th in zip(lambdas, thetas):
        total += lam * (family.A(th + t) - family.A(th))
    return total


def compute_aggregate_mean(
    family: EDMFamily,
    lambdas: np.ndarray,
    thetas: np.ndarray,
) -> float:
    """Expected value of aggregate S.

    E[S] = K_S'(0) = sum_i lambda_i * A'(theta_i)

    Parameters
    ----------
    family : EDMFamily
        An instantiated EDM family.
    lambdas, thetas : np.ndarray
        EDM parameters; shape (n,).

    Returns
    -------
    float
        E[S].
    """
    lambdas = np.asarray(lambdas, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    return float(np.sum([lam * family.A_prime(th) for lam, th in zip(lambdas, thetas)]))


def _ks_prime(
    family: EDMFamily,
    t: float,
    lambdas: np.ndarray,
    thetas: np.ndarray,
) -> float:
    """K_S'(t) = sum_i lambda_i * A'(theta_i + t)."""
    return float(np.sum([lam * family.A_prime(th + t) for lam, th in zip(lambdas, thetas)]))


def compute_saddle_point(
    family: EDMFamily,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    s: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Solve K_S'(t_hat) = s for the saddle-point tilt t_hat.

    K_S'(t) = sum_i lambda_i * A'(theta_i + t)

    Since A is strictly convex, K_S' is strictly increasing in t.  The
    function uses ``scipy.optimize.brentq`` after automatically bracketing
    the root.

    Parameters
    ----------
    family : EDMFamily
        An instantiated EDM family.
    lambdas, thetas : np.ndarray
        EDM parameters; shape (n,).
    s : float
        Target value of K_S'(t_hat).  Must satisfy:

            K_S'(theta_lo - max(thetas)) < s < K_S'(theta_hi - min(thetas))

        i.e. s must be in the interior of the range of E[S | tilt].
    tol : float
        Tolerance for brentq.
    max_iter : int
        Maximum iterations for brentq.

    Returns
    -------
    float
        t_hat such that K_S'(t_hat) = s.

    Raises
    ------
    ValueError
        If s is outside the achievable range or bracketing fails.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    theta_lo, theta_hi = family.theta_domain()

    # The constraint is theta_i + t must remain inside (theta_lo, theta_hi)
    # => t in (theta_lo - max(thetas), theta_hi - min(thetas))
    t_lo = theta_lo - float(np.max(thetas))   # lower bound for t
    t_hi = theta_hi - float(np.min(thetas))   # upper bound for t

    # Find a finite bracket by searching from the natural point t=0
    _EPS = 1e-8
    _EXPAND = 10.0

    # Compute the mean at t=0 (no tilt)
    mean_s = _ks_prime(family, 0.0, lambdas, thetas)

    if np.isclose(s, mean_s, rtol=1e-10, atol=1e-14):
        return 0.0

    # Determine search direction
    if s > mean_s:
        # Need t > 0; search upward toward t_hi
        t_a = 0.0
        t_b = min(1.0, 0.5 * t_hi) if np.isfinite(t_hi) else 1.0
        for _ in range(max_iter):
            try:
                val_b = _ks_prime(family, t_b, lambdas, thetas)
            except (ValueError, RuntimeError):
                t_b /= 2.0
                continue
            if val_b >= s:
                break
            t_a = t_b
            # Expand toward t_hi
            step = t_b - t_a + _EPS
            t_b = min(t_b + step * _EXPAND, t_hi - _EPS) if np.isfinite(t_hi) else t_b + step * _EXPAND
        else:
            raise ValueError(
                f"Could not bracket saddle point above mean: s={s}, mean={mean_s}."
            )
    else:
        # Need t < 0; search downward toward t_lo
        t_b = 0.0
        t_a = max(-1.0, 0.5 * t_lo) if np.isfinite(t_lo) else -1.0
        for _ in range(max_iter):
            try:
                val_a = _ks_prime(family, t_a, lambdas, thetas)
            except (ValueError, RuntimeError):
                t_a /= 2.0
                continue
            if val_a <= s:
                break
            t_b = t_a
            step = abs(t_b - t_a) + _EPS
            t_a = max(t_a - step * _EXPAND, t_lo + _EPS) if np.isfinite(t_lo) else t_a - step * _EXPAND
        else:
            raise ValueError(
                f"Could not bracket saddle point below mean: s={s}, mean={mean_s}."
            )

    def equation(t: float) -> float:
        return _ks_prime(family, t, lambdas, thetas) - s

    try:
        t_hat = optimize.brentq(
            equation,
            t_a,
            t_b,
            xtol=tol,
            maxiter=max_iter,
            full_output=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"brentq failed to find saddle point for s={s}: {exc}"
        ) from exc

    return float(t_hat)


def compute_g_hat_i(
    family: EDMFamily,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    s: float,
    i: int,
) -> float:
    """Saddle-point conditional share proxy for component i at aggregate level s.

    The proxy is defined as:

        g_hat_i(s) = lambda_i * A'(theta_i + t_hat(s))
                     / sum_j lambda_j * A'(theta_j + t_hat(s))

    where t_hat(s) is the saddle-point tilt solving K_S'(t_hat) = s.

    Under the tilted measure, the conditional mean of X_i given S = s is
    approximately lambda_i * A'(theta_i + t_hat), so g_hat_i is the
    approximate conditional share.

    Parameters
    ----------
    family : EDMFamily
        An instantiated EDM family.
    lambdas, thetas : np.ndarray
        EDM parameters; shape (n,).
    s : float
        Target aggregate value.
    i : int
        Component index (0-based).

    Returns
    -------
    float
        g_hat_i(s) in (0, 1).
    """
    lambdas = np.asarray(lambdas, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    t_hat = compute_saddle_point(family, lambdas, thetas, s)

    numerator = lambdas[i] * family.A_prime(thetas[i] + t_hat)
    denominator = float(
        np.sum([lam * family.A_prime(th + t_hat) for lam, th in zip(lambdas, thetas)])
    )

    if abs(denominator) < 1e-15:
        warnings.warn(
            f"Denominator of g_hat_i is near zero at s={s}; returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    return float(numerator / denominator)
