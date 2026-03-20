# HW4 Experiments

This directory contains the Homework 4 experiment scripts, plots, and results. It expects the shared `bandit_sim` library from the repository root to already be installed.

## Structure

```text
part1.py
part2.py
```

## Design

- The shared library code is maintained outside `hw4` in `../src/bandit_sim`.
- `part1.py` runs the Bayesian regret experiment for Thompson Sampling under a well-specified Gaussian prior and a misspecified uniform prior.
- `part2.py` runs the frequentist regret experiment comparing UCB and Thompson Sampling on a fixed 10-armed Gaussian bandit family.
- Environment setup is documented in the repository root `README.md`.

## Part I: Bayesian Regret

Run Part I with:

```bash
python part1.py
```

This experiment:

- samples a fresh 10-armed Gaussian bandit from the prior \(P = \mathcal{N}(0_K, I_K)\) for each simulation
- runs Thompson Sampling under:
  - the true Gaussian prior
  - the misspecified uniform prior on `[-1, 1]^K`
- evaluates horizons `100`, `1000`, and `10000`
- runs `50` simulations per horizon

It generates:

- `results/part1_bayes_regret.png`: Bayes regret versus horizon with error bars
- `results/part1_bayes_regret.json`: empirical Bayes regret and standard error for each horizon and prior assumption

## Part II: Frequentist Regret

Run Part II with:

```bash
python part2.py
```

This experiment:

- uses a fixed 10-armed Gaussian bandit with:
  - arm 1 mean `0.5`
  - arm 2 mean `0.5 - Δ`
  - arms 3-10 mean `-0.5`
- sets reward variance to `1`
- uses horizon `2000`
- sweeps \(Δ ∈ \{0.05, 0.1, ..., 1.0\}\)
- runs `10` simulations per delta
- compares:
  - `UCB`
  - `Thompson Sampling` with the Gaussian prior from Part I

It generates:

- `results/part2_frequentist_regret.png`: empirical regret versus delta with error bars
- `results/part2_frequentist_regret.json`: empirical average regret and standard error for each delta and algorithm
