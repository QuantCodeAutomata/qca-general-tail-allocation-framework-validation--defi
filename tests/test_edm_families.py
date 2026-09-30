"""Tests for src/edm_families.py — EDM family implementations."""

import numpy as np
import pytest
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.edm_families import (
    NormalEDM, PoissonEDM, ScaledPoissonEDM, GammaEDM,
    TweedieEDM, GenHypSecantEDM,
    compute_saddle_point, compute_g_hat_i, compute_aggregate_mean,
    compute_cumulant_aggregate,
)

RNG = np.random.default_rng(2025)

# ---------------------------------------------------------------
# NormalEDM
# ---------------------------------------------------------------

class TestNormalEDM:
    def setup_method(self):
        self.fam = NormalEDM()

    def test_A_theta(self):
        """A(theta) = theta^2/2."""
        for theta in [-2.0, 0.0, 1.0, 3.0]:
            assert abs(self.fam.A(theta) - theta**2 / 2) < 1e-12

    def test_A_prime(self):
        """A'(theta) = theta."""
        for theta in [-1.0, 0.0, 2.0]:
            assert abs(self.fam.A_prime(theta) - theta) < 1e-12

    def test_A_double_prime(self):
        """A''(theta) = 1."""
        for theta in [-1.0, 0.0, 2.0]:
            assert abs(self.fam.A_double_prime(theta) - 1.0) < 1e-12

    def test_theta_domain_is_real_line(self):
        """Normal: theta in (-inf, inf)."""
        lo, hi = self.fam.theta_domain()
        assert lo < -1e10
        assert hi > 1e10

    def test_exact_g_i(self):
        """exact_g_i satisfies the linear conditional share formula."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        mu_s = mus.sum()
        sig_s = np.sqrt((sigmas**2).sum())
        beta1 = sigmas[0]**2 / sig_s**2

        for s in [1.5, 2.0, 3.5, 5.0]:
            g1 = self.fam.exact_g_i(s, 0, mus, sigmas)
            expected = beta1 + (mus[0] - beta1 * mu_s) / s
            assert abs(g1 - expected) < 1e-10

    def test_exact_g_i_sums_to_one(self):
        """g_1(s) + g_2(s) = 1 for all s."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        for s in [0.5, 1.0, 2.0, 4.0]:
            g1 = self.fam.exact_g_i(s, 0, mus, sigmas)
            g2 = self.fam.exact_g_i(s, 1, mus, sigmas)
            assert abs(g1 + g2 - 1.0) < 1e-10

    def test_aggregate_cdf_ppf_consistency(self):
        """aggregate_cdf(ppf(u)) = u."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        for u in [0.1, 0.5, 0.9]:
            x = self.fam.aggregate_ppf(u, mus, sigmas)
            u_back = self.fam.aggregate_cdf(x, mus, sigmas)
            assert abs(u_back - u) < 1e-8

    def test_sample_aggregate_mean_std(self):
        """Sampled aggregate has correct mean and std."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        lambdas = sigmas**2
        thetas = mus / sigmas**2
        s = self.fam.sample_aggregate(lambdas, thetas, 100_000, RNG)
        mu_s = mus.sum()
        sig_s = np.sqrt((sigmas**2).sum())
        assert abs(np.mean(s) - mu_s) < 0.05
        assert abs(np.std(s) - sig_s) < 0.05

# ---------------------------------------------------------------
# PoissonEDM
# ---------------------------------------------------------------

class TestPoissonEDM:
    def setup_method(self):
        self.fam = PoissonEDM()

    def test_A_is_exp(self):
        """A(theta) = exp(theta)."""
        for theta in [-1.0, 0.0, 1.0, 2.0]:
            assert abs(self.fam.A(theta) - np.exp(theta)) < 1e-12

    def test_iota_is_constant_one(self):
        """iota(theta) = A'(theta)/A''(theta) = exp(theta)/exp(theta) = 1."""
        for theta in [-1.0, 0.0, 1.0]:
            assert abs(self.fam.iota(theta) - 1.0) < 1e-8

    def test_equality_weights_homogeneous(self):
        """Homogeneous theta: r_i = lambda_i / sum(lambda_j)."""
        lambdas = np.array([2.0, 3.0])
        r = self.fam.equality_weights_homogeneous(lambdas)
        expected = lambdas / lambdas.sum()
        np.testing.assert_allclose(r, expected, atol=1e-12)

    def test_equality_weights_heterogeneous(self):
        """Heterogeneous: r_i = lambda_i*exp(theta_i) / sum."""
        lambdas = np.array([2.0, 3.0])
        thetas = np.array([0.5, 1.0])
        r = self.fam.equality_weights_heterogeneous(lambdas, thetas)
        weights = lambdas * np.exp(thetas)
        expected = weights / weights.sum()
        np.testing.assert_allclose(r, expected, atol=1e-12)

    def test_equality_weights_sum_to_one(self):
        """Weights sum to 1."""
        lambdas = np.array([1.0, 2.0, 3.0])
        thetas = np.array([0.0, 0.5, 1.0])
        r = self.fam.equality_weights_heterogeneous(lambdas, thetas)
        assert abs(r.sum() - 1.0) < 1e-12

# ---------------------------------------------------------------
# ScaledPoissonEDM
# ---------------------------------------------------------------

class TestScaledPoissonEDM:
    def test_iota_is_1_over_b(self):
        """iota(theta) = 1/b for scaled Poisson."""
        for b in [0.5, 1.0, 2.0]:
            fam = ScaledPoissonEDM(a=1.0, b=b)
            for theta in [-1.0, 0.0, 1.0]:
                assert abs(fam.iota(theta) - 1.0 / b) < 1e-8

    def test_equality_weights_formula(self):
        """r_i = lambda_i exp(b theta_i) / sum."""
        fam = ScaledPoissonEDM(a=1.0, b=2.0)
        lambdas = np.array([1.0, 2.0])
        thetas = np.array([0.5, 1.0])
        r = fam.equality_weights(lambdas, thetas)
        weights = lambdas * np.exp(2.0 * thetas)
        expected = weights / weights.sum()
        np.testing.assert_allclose(r, expected, atol=1e-12)

    def test_reduces_to_poisson_for_a_b_1(self):
        """ScaledPoisson(a=1, b=1) should have same A as PoissonEDM."""
        sp = ScaledPoissonEDM(a=1.0, b=1.0)
        pois = PoissonEDM()
        for theta in [-0.5, 0.0, 0.5]:
            assert abs(sp.A(theta) - pois.A(theta)) < 1e-6

# ---------------------------------------------------------------
# GammaEDM
# ---------------------------------------------------------------

class TestGammaEDM:
    def setup_method(self):
        self.fam = GammaEDM()

    def test_A_is_neg_log_neg_theta(self):
        """A(theta) = -log(-theta) for theta < 0."""
        for theta in [-3.0, -1.0, -0.5, -0.1]:
            expected = -np.log(-theta)
            assert abs(self.fam.A(theta) - expected) < 1e-12

    def test_A_prime(self):
        """A'(theta) = -1/theta."""
        for theta in [-3.0, -1.0, -0.5]:
            assert abs(self.fam.A_prime(theta) - (-1.0 / theta)) < 1e-12

    def test_A_double_prime(self):
        """A''(theta) = 1/theta^2."""
        for theta in [-3.0, -1.0, -0.5]:
            assert abs(self.fam.A_double_prime(theta) - 1.0 / theta**2) < 1e-12

    def test_iota_equals_neg_theta(self):
        """iota(theta) = A'/A'' = (-1/theta)/(1/theta^2) = -theta > 0 for theta < 0."""
        for theta in [-3.0, -1.0, -0.5]:
            expected = -theta
            assert abs(self.fam.iota(theta) - expected) < 1e-8, \
                f"theta={theta}: iota={self.fam.iota(theta):.6f}, expected={expected:.6f}"

    def test_g_i_homogeneous_constant(self):
        """With equal theta, conditional share is constant = lambda_i/sum(lambda_j)."""
        lambdas = np.array([2.0, 3.0])
        theta = -1.0
        for s in [1.0, 2.0, 5.0]:
            g1 = self.fam.g_i_exact_homogeneous(s, 0, lambdas, theta)
            expected = lambdas[0] / lambdas.sum()
            assert abs(g1 - expected) < 1e-12

    def test_sample_components_positive(self):
        """Gamma samples are all positive."""
        lambdas = np.array([2.0, 3.0])
        thetas = np.array([-1.0, -0.5])
        X = self.fam.sample_components(lambdas, thetas, 1000, RNG)
        assert np.all(X > 0)

# ---------------------------------------------------------------
# TweedieEDM
# ---------------------------------------------------------------

class TestTweedieEDM:
    def test_p0_reduces_to_normal(self):
        """Tweedie p=0 is Normal: A(theta) = theta^2/2."""
        fam = TweedieEDM(p=0)
        normal = NormalEDM()
        for theta in [-1.0, 0.0, 1.0]:
            assert abs(fam.A(theta) - normal.A(theta)) < 1e-8

    def test_p2_reduces_to_gamma(self):
        """Tweedie p=2 is Gamma: A(theta) = -log(-theta)."""
        fam = TweedieEDM(p=2)
        gamma = GammaEDM()
        for theta in [-3.0, -1.0, -0.5]:
            assert abs(fam.A(theta) - gamma.A(theta)) < 1e-6, \
                f"theta={theta}: Tweedie={fam.A(theta):.6f}, Gamma={gamma.A(theta):.6f}"

    def test_iota_general(self):
        """iota(theta) = A'/A'' = (1-p)*theta for general Tweedie."""
        for p in [0.0, 1.5, 3.0]:  # p=0 (Normal), p=1.5 (compound Poisson-Gamma), p=3.0
            fam = TweedieEDM(p=p)
            for theta in [-2.0, -1.0, -0.5]:
                iota_val = fam.iota(theta)
                # For p != 1, 2: iota = (1-p)*theta
                if p != 1 and p != 2:
                    expected = (1 - p) * theta
                    assert abs(iota_val - expected) < 1e-6, \
                        f"p={p}, theta={theta}: iota={iota_val:.4f}, expected={expected:.4f}"

    def test_A_prime_positive_for_compound_poisson(self):
        """A'(theta) > 0 for compound Poisson-Gamma (1 < p < 2)."""
        fam = TweedieEDM(p=1.5)
        for theta in [-3.0, -1.0, -0.5]:
            assert fam.A_prime(theta) > 0

# ---------------------------------------------------------------
# GenHypSecantEDM
# ---------------------------------------------------------------

class TestGenHypSecantEDM:
    def setup_method(self):
        self.fam = GenHypSecantEDM()

    def test_A_theta(self):
        """A(theta) = -log(cos(theta))."""
        for theta in [-0.5, 0.0, 0.5, 1.0]:
            expected = -np.log(np.cos(theta))
            assert abs(self.fam.A(theta) - expected) < 1e-10

    def test_A_prime_is_tan(self):
        """A'(theta) = tan(theta)."""
        for theta in [-0.5, 0.0, 0.3, 0.7]:
            assert abs(self.fam.A_prime(theta) - np.tan(theta)) < 1e-10

    def test_iota_formula(self):
        """iota(theta) = sin(2*theta)/2 = sin(theta)*cos(theta)."""
        for theta in [-0.5, 0.0, 0.3, 0.7]:
            expected = np.sin(theta) * np.cos(theta)
            assert abs(self.fam.iota(theta) - expected) < 1e-8

    def test_switching_tilt(self):
        """t* = pi/4 - theta_min."""
        thetas = np.array([0.2, 0.6])
        t_star = self.fam.switching_tilt(thetas)
        expected = np.pi / 4 - thetas.min()
        assert abs(t_star - expected) < 1e-10

    def test_theta_domain(self):
        """theta domain is (-pi/2, pi/2)."""
        lo, hi = self.fam.theta_domain()
        assert abs(lo - (-np.pi / 2)) < 1e-10
        assert abs(hi - np.pi / 2) < 1e-10

# ---------------------------------------------------------------
# Saddle-point utilities
# ---------------------------------------------------------------

class TestSaddlePoint:
    def setup_method(self):
        self.normal_fam = NormalEDM()
        self.gamma_fam = GammaEDM()
        self.lambdas = np.array([2.0, 3.0])

    def test_saddle_point_normal(self):
        """For Normal, K_S'(t) = sum lambda_i * A'(theta_i + t) = sum lambda_i*(theta_i+t) = s."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        lambdas = sigmas**2
        thetas = mus / sigmas**2
        mu_s = mus.sum()
        sig_s = np.sqrt((sigmas**2).sum())

        # K_S'(t) = mu_s + sig_s^2 * t = s => t = (s - mu_s) / sig_s^2
        for s in [2.0, 3.5, 5.0]:
            t_hat = compute_saddle_point(self.normal_fam, lambdas, thetas, s)
            expected = (s - mu_s) / sig_s**2
            assert abs(t_hat - expected) < 1e-6, \
                f"s={s}: t_hat={t_hat:.6f}, expected={expected:.6f}"

    def test_g_hat_sums_to_one(self):
        """sum_i g_hat_i(s) = 1 for any s."""
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        lambdas = sigmas**2
        thetas = mus / sigmas**2

        for s in [2.0, 3.0, 5.0]:
            g0 = compute_g_hat_i(self.normal_fam, lambdas, thetas, s, 0)
            g1 = compute_g_hat_i(self.normal_fam, lambdas, thetas, s, 1)
            assert abs(g0 + g1 - 1.0) < 1e-8

    def test_g_hat_normal_equals_exact(self):
        """For Normal, saddle-point g_hat equals exact g_i."""
        fam = NormalEDM()
        mus = np.array([1.0, 2.0])
        sigmas = np.array([1.0, 1.5])
        lambdas = sigmas**2
        thetas = mus / sigmas**2
        mu_s = mus.sum()
        sig_s = np.sqrt((sigmas**2).sum())

        for s in [2.5, 3.5, 5.0]:
            g_hat = compute_g_hat_i(fam, lambdas, thetas, s, 0)
            g_exact = fam.exact_g_i(s, 0, mus, sigmas)
            assert abs(g_hat - g_exact) < 1e-6, \
                f"s={s}: g_hat={g_hat:.6f}, g_exact={g_exact:.6f}"

    def test_compute_cumulant_aggregate_zero_at_zero(self):
        """K_S(0) = 0 by definition (normalization)."""
        fam = NormalEDM()
        lambdas = np.array([1.0, 2.0])
        thetas = np.array([0.5, 1.0])
        val = compute_cumulant_aggregate(fam, 0.0, lambdas, thetas)
        assert abs(val) < 1e-12

    def test_aggregate_mean_formula(self):
        """E[S] = sum_i lambda_i A'(theta_i)."""
        fam = NormalEDM()
        lambdas = np.array([1.0, 2.0])
        thetas = np.array([0.5, 1.0])
        # For Normal: A'(theta) = theta, so E[S] = sum lambda_i theta_i
        expected = np.sum(lambdas * thetas)
        result = compute_aggregate_mean(fam, lambdas, thetas)
        assert abs(result - expected) < 1e-10

# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

def test_normal_sample_components_shape():
    """sample_components returns (n_comp, n_mc) — components are rows."""
    fam = NormalEDM()
    lambdas = np.array([1.0, 2.0, 3.0])
    thetas = np.array([0.5, 1.0, -0.5])
    X = fam.sample_components(lambdas, thetas, 1000, RNG)
    assert X.shape == (3, 1000)

def test_gamma_aggregate_cdf_n2_homogeneous():
    """Homogeneous Gamma n=2: CDF matches scipy gamma."""
    fam = GammaEDM()
    lambda1, lambda2 = 2.0, 3.0
    theta = -1.0
    # S ~ Gamma(lambda1+lambda2, -theta) = Gamma(5, 1)
    for s in [2.0, 5.0, 10.0]:
        val = fam.aggregate_cdf_n2(s, lambda1, lambda2, theta, theta)
        expected = stats.gamma.cdf(s, a=lambda1 + lambda2, scale=1.0 / (-theta))
        assert abs(val - expected) < 0.01, \
            f"s={s}: val={val:.4f}, expected={expected:.4f}"
