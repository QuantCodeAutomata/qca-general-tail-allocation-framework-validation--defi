# Repository Knowledge (AGENTS.md)

## Project Overview
Nonparametric tail-allocation framework validation.  Four experiment scripts
(exp_1 … exp_4) import from `src/` and write outputs to `results/`.

## `src/tail_allocation.py` — Core Module

### Public API (all functions and the dataclass)

| Symbol | Description |
|--------|-------------|
| `compute_left_quantile(cdf_func, u, x_min, x_max, tol)` | q(u) = inf{x:F(x)>=u} via brentq |
| `compute_quantile_grid(cdf_func, u_grid, x_min, x_max)` | vectorised q(u) |
| `compute_es(q_grid, u_grid)` | ES_u = (1/(1-u)) ∫_u^1 q(v)dv — trapezoidal with tail correction |
| `compute_W_u(es_grid, q_grid)` | W_u = ES_u / q(u); np.nan where q=0 |
| `compute_theta_u(cdf_s, pmf_s, q_u, u)` | θ_u = (F(q(u))-u)/P(S=q(u)); 0 when pmf_s=None |
| `compute_pi_u(s_values, q_u, theta_u)` | π_u(s) = 1_{s>q_u} + θ_u 1_{s=q_u} |
| `compute_phi_continuous(x_i, s, q_u, theta_u)` | MC Φ_{i,u} = E[X_i π_u]/E[S π_u] |
| `compute_psi_continuous(r_i, s, q_u, theta_u)` | MC Ψ_{i,u} = E[R_i π_u]/E[π_u] |
| `compute_r_i(x_i, s, c_i)` | R_i = X_i/S on {S≠0}, c_i elsewhere |
| `compute_phi_quantile(q_xi_func, q_s_func, u, u_max, n_points)` | Φ via quantile quadrature (comonotonic) |
| `compute_psi_quantile(q_xi_func, q_s_func, u, u_max, n_points)` | Ψ via quantile quadrature |
| `compute_h(q_xi_func, q_s_func, u_grid)` | h(u) = q_Xi(u)/q_S(u); np.nan where q_S=0 |
| `compute_covariance_identity(q_s_func, h_func, u, u_max, n_points)` | verify Φ-Ψ = Cov_u(q_S,h)/E_u[q_S] |
| `compute_effective_tail_domain(cdf_s, q_u_star, u_star, s_max)` | lower_open/lower/upper of tail support |
| `validate_allocations(phi_list, psi_list, tol)` | checks Σ Φ = 1, Σ Ψ = 1 |
| `TailAllocationScenario` | dataclass: name, u/q/es/W grids, phi/psi/delta/h_results dicts; `.to_dataframe()` |

### Key Design Decisions
- **compute_left_quantile**: uses `scipy.optimize.brentq`; discrete CDFs may
  return an interior value (use `compute_theta_u` to correct).
- **compute_es**: O(n) via right-cumulative trapezoidal sum; adds tail
  correction `q[-1] * (1 - u[-1])` for grids that don't reach u=1.
- **compute_r_i**: avoids `RuntimeWarning` by replacing the denominator with 1
  on `{S=0}` events and then overwriting with `c_i`.
- **compute_phi_quantile vs MC**: quantile formula equals the MC formula only
  when X_i and S are comonotonic.  For independent risks use MC methods.
- **compute_covariance_identity**: normalises by `width = u_max - u`; since
  Cov and E_u[q_S] share the same normalisation, the ratio equals Φ-Ψ exactly.
- **compute_effective_tail_domain**: `lower_open = (F_S(q(u*)) ≈ u*)` — open
  for continuous distributions, closed when there is a genuine atom at q(u*).

### Testing Pattern
```python
from scipy import stats
from src.tail_allocation import *
cdf = lambda x: stats.norm.cdf(x)
u_grid = np.linspace(0.5, 0.999, 500)
q_grid = compute_quantile_grid(cdf, u_grid, -10, 10)
es_grid = compute_es(q_grid, u_grid)
```

## `src/asymptotic.py` — Asymptotic Tail-Equivalence Analysis

### Public API

| Symbol | Description |
|--------|-------------|
| `compute_W_sequence(q_func, es_func, u_seq)` | W_u = ES_u/q_u along a sequence; NaN where \|q\| < 1e-15 |
| `compute_phi_psi_sequence(phi_func, psi_func, u_seq)` | Evaluate Phi and Psi along u_k -> 1 |
| `estimate_convergence(values, window=10)` | limsup/liminf/oscillation_amplitude/converged over last `window` values |
| `check_theorem_bounds(phi_seq, psi_seq, W_seq, tol)` | Check \|Psi-Phi\| <= 2*(W-1)*sup\|Phi\|; returns bound_holds/residuals/max_violation |
| `ParetoBoundedWScenario(alphas, sigmas)` | Comonotonic Pareto: analytic q_s/es_s/W_u/phi_i/psi_i/asymptotic_limit; alpha_i > 1 |
| `OscillatingShareScenario(amplitude, frequency)` | S~Exp(1), g_1(s)=0.5+A*sin(omega*s); W_u->1, Phi/Psi oscillate |
| `DiscreteDivergentWScenario(N, alpha_even, alpha_odd)` | S at {2^k} with alternating shares; Phi->0.5, Psi oscillates |
| `compute_asymptotic_diagnostics(phi, psi, W, u)` | DataFrame with u/phi/psi/W/delta/phi_osc/psi_osc/classification |
| `classify_asymptotic_behavior(phi, psi, W)` | Returns 'common_limit', 'phi_converges_psi_oscillates', or 'delta_vanishes_no_limit' |

### Key Design Decisions
- **psi_i (Pareto)**: uses change-of-variables t=1-v and factors out t^{-1/alpha_min} for numerical stability; all remaining power exponents are non-negative.
- **psi_i (Oscillating)**: analytic closed form via Laplace-transform:
  `Psi = 0.5 + A*(omega*cos(omega*q_u) + sin(omega*q_u)) / (1 + omega^2)`.
- **phi_i (Pareto)**: fully analytic using int_u^1 (1-v)^{-1/a} dv = a/(a-1)*(1-u)^{1-1/a}.
- **DiscreteDivergentWScenario._theta_u**: exact formula (F_S(q_u)-u)/P(S=q_u), clamped to [0,1]; called internally by es_s, phi_i, psi_i.
- **classify_asymptotic_behavior**: decision order — (1) both converged => common_limit; (2) only phi converged => phi_converges_psi_oscillates; (3) else => delta_vanishes_no_limit.
- **_running_oscillation**: O(n*window) helper used only inside compute_asymptotic_diagnostics; not exported.

## `src/edm_families.py` — Additive EDM Families

Framework: X_i ~ EDM(lambda_i, theta_i, A) with CGF
K_{X_i}(t) = lambda_i * [A(theta_i + t) - A(theta_i)].

### Class Hierarchy

| Class | A(theta) | theta domain | Notes |
|-------|----------|--------------|-------|
| `EDMFamily` | abstract | — | base; `iota = A'/A''` default |
| `NormalEDM` | theta^2/2 | R | lambda=sigma^2, theta=mu/sigma^2 |
| `PoissonEDM` | exp(theta) | R | iota=1 |
| `ScaledPoissonEDM(a,b)` | (a/b)*exp(b*theta) | R | a*b>0; iota=1/b |
| `GammaEDM` | -log(-theta) | (-inf,0) | iota=-theta |
| `TweedieEDM(p)` | power-p formula | R (p=0,1), (-inf,0) (p>=1) | p in {0,1,2,3} or 1<p<2 |
| `GenHypSecantEDM` | -log(cos(theta)) | (-pi/2,pi/2) | iota=sin*cos=sin(2theta)/2 |

### Tweedie formula (p≠0,1,2): general case
- kappa = (2-p)/(1-p)
- A(theta) = [(1-p)*theta]^kappa / (2-p), theta<0 for p>1
- A'(theta) = [(1-p)*theta]^{kappa-1}
- A''(theta) = (kappa-1)*(1-p)*[(1-p)*theta]^{kappa-2}
- iota(theta) = (1-p)*theta

Special cases verified: p=3 gives A=-sqrt(-2*theta); p=1.5 gives A=4/(-theta) at theta=-1.

### Key iota values
- Normal p=0: iota(theta)=theta
- Poisson p=1: iota=1
- Gamma p=2: iota=-theta (NOT -1/theta — the spec has a typo)
- Tweedie p=3: iota=(1-3)*theta=-2*theta
- GenHypSecant: iota=sin(theta)*cos(theta)

### Module-level utilities

| Function | Description |
|----------|-------------|
| `compute_cumulant_aggregate(family, t, lambdas, thetas)` | K_S(t)=sum lambda_i*(A(theta_i+t)-A(theta_i)) |
| `compute_aggregate_mean(family, lambdas, thetas)` | E[S]=sum lambda_i*A'(theta_i) |
| `compute_saddle_point(family, lambdas, thetas, s)` | Solve K_S'(t_hat)=s via brentq with auto-bracketing |
| `compute_g_hat_i(family, lambdas, thetas, s, i)` | g_hat_i=lambda_i*A'(theta_i+t_hat)/sum lambda_j*A'(theta_j+t_hat) |

### Design decisions
- **NormalEDM.iota**: returns theta (identity). **GammaEDM.iota**: returns -theta (positive since theta<0).
- **g_hat_i can be negative** for Normal when component has negative mean — this is correct (sum=1 is the invariant).
- **Saddle point bracketing**: starts at t=0 (mean), expands geometrically toward theta domain boundaries.
- **TweedieEDM compound Poisson-Gamma (1<p<2)**: uses N~Poisson(lambda_i*A(theta_i)) Gamma(r,-theta_i) jumps where r=(2-p)/(p-1).
- **GenHypSecantEDM.sample_unit**: inversion-CDF for theta=0 (F^{-1}(u)=log(tan(pi*u/2))), then SIR reweighting for theta≠0.
- **ScaledPoissonEDM sampling**: Y=b*X~Poisson(lambda_i*a*exp(b*theta_i)), X=Y/b.
- **GammaEDM.aggregate_cdf_n2**: exact Gamma CDF for homogeneous thetas; numerical convolution integral for heterogeneous.

## Dependencies
`numpy`, `scipy`, `pandas` — install with `pip install numpy scipy pandas`.

## Running
```bash
python src/tail_allocation.py        # self-test
pytest tests/ -v                     # full test suite
python exp/exp_1_framework.py        # experiment 1
```
