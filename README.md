# General Tail-Allocation Framework Validation

Nonparametric tail-allocation framework validation covering realized risk shares,
effective tail domains, proportional ES/GTE allocation, comonotonic risks,
independent additive EDMs, and asymptotic tail-equivalence via W_u.

## Repository Structure

```
├── data/          # Scenario configuration files
├── src/           # Core implementation modules
│   ├── tail_allocation.py   # Core framework: q(u), ES, pi_u, Phi, Psi, g_i
│   ├── edm_families.py      # Additive EDM families (Normal, Poisson, Gamma, Tweedie, GHS)
│   ├── comonotonic.py       # Comonotonic allocation formulas
│   └── asymptotic.py        # Asymptotic W_u analysis and counterexamples
├── exp/           # Experiment scripts
│   ├── exp_1_framework.py   # Exp 1: Core framework validation
│   ├── exp_2_edm.py         # Exp 2: Independent additive EDM experiments
│   ├── exp_3_comonotonic.py # Exp 3: Comonotonic risk experiments
│   └── exp_4_asymptotic.py  # Exp 4: Asymptotic tail-equivalence experiments
├── tests/         # pytest test suite
├── results/       # Output tables, plots, and RESULTS.md
```

## Setup

```bash
pip install numpy scipy pandas matplotlib
```

## Running Experiments

```bash
python exp/exp_1_framework.py
python exp/exp_2_edm.py
python exp/exp_3_comonotonic.py
python exp/exp_4_asymptotic.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Mathematical Objects

- `q(u)` = VaR_u[S]: left-continuous quantile of aggregate S
- `ES_u(S)` = (1/(1-u)) ∫_u^1 q(v) dv: Expected Shortfall
- `W_u` = ES_u(S)/q(u): tail-to-quantile multiplier
- `pi_u(s)` = 1_{s>q(u)} + θ_u 1_{s=q(u)}: randomized tail indicator
- `Phi_{i,u}` = E[X_i π_u(S)] / E[S π_u(S)]: proportional ES allocation
- `Psi_{i,u}` = E[R_i π_u(S)] / E[π_u(S)]: proportional GTE allocation
- `g_i(s)` = E[R_i | S=s]: conditional share function
