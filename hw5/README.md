# HW5 Experiments

This directory contains the Homework 5 experiment scripts, plots, and results. It expects the shared `bandit_sim` library from the repository root to already be installed.

## Structure

```text
part_a.py
part_b.py
part_c.py
part_e.py
```

## Design

- The shared library code is maintained outside `hw5` in `../src/bandit_sim`.
- All experiments use a 2-armed Bernoulli bandit with arm means μ₁ = 0.5 and μ₂ = μ₁ + Δ.
- `part_a.py` compares UCB (β = 0.25 and β = 1.0) and EXP3 across a range of horizon values.
- `part_b.py` sweeps EXP3 learning rates at a fixed horizon for a fixed Δ.
- `part_c.py` repeats the learning rate sweep of Part B across several Δ values.
- `part_e.py` repeats the learning rate sweep of Part C for a large gap value Δ = 0.5.
- Each script uses a single `master_rng` seeded from `master_seed`. Independent integer seeds for bandits and algorithms are drawn from it via `draw_seed(rng)`, ensuring reproducibility and avoiding seed collisions.
- Environment setup is documented in the repository root `README.md`.

## Part A: Horizon Sweep

Run Part A with:

```bash
python part_a.py
```

This experiment:

- uses a 2-armed Bernoulli bandit with μ₁ = 0.5, μ₂ = 0.55
- sweeps horizons `10`, `100`, `1000`, `10000`, `100000`
- runs `100` simulations per horizon
- compares UCB (β = 0.25), UCB (β = 1.0), and EXP3, where EXP3 uses the theoretical learning rate η = √(2 ln K / (T K)) for each horizon

It generates:

- `results/regret_vs_horizon.png`: empirical average regret versus horizon for UCB and EXP3
- `results/regret_vs_horizon.json`: empirical average regret and standard error for each horizon and algorithm

## Part B: Learning Rate Sweep

Run Part B with:

```bash
python part_b.py [--n-simulations N]
```

> **Note:** At 100 or more simulations this script can take several minutes to complete.

This experiment:

- uses a 2-armed Bernoulli bandit with μ₁ = 0.5, μ₂ = 0.55
- fixes horizon at `100000`
- sweeps learning rates η ∈ {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1} plus the theoretical optimum η = √(2 ln K / (T K))
- runs `100` simulations per learning rate (overridable via `--n-simulations`)

It generates:

- `results/exp3_lr_sweep.png`: empirical average regret versus learning rate
- `results/exp3_lr_sweep.json`: empirical average regret and standard error for each learning rate

## Part C: Learning Rate Sweep Across Deltas

Run Part C with:

```bash
python part_c.py [--n-simulations N]
```

> **Note:** Part C runs sequentially over all (Δ, η) combinations. At 100 or more simulations this script can take a long time to complete.

This experiment:

- uses a 2-armed Bernoulli bandit with μ₁ = 0.5
- sweeps Δ ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.25}, so μ₂ = μ₁ + Δ
- for each Δ, sweeps the same learning rates as Part B
- runs `100` simulations per (Δ, η) combination (overridable via `--n-simulations`)
- produces a 2×3 grid of subplots, one per Δ value

It generates:

- `results/exp3_delta_lr_sweep.png`: 2×3 grid of regret-vs-learning-rate plots across Δ values
- `results/exp3_delta_lr_sweep.json`: empirical average regret and standard error for each (Δ, η) combination

## Part E: Learning Rate Sweep for Large Deltas

Run Part E with:

```bash
python part_e.py [--n-simulations N]
```

> **Note:** Part E runs sequentially over all (Δ, η) combinations. At 100 or more simulations this script can take a long time to complete.

This experiment:

- uses a 2-armed Bernoulli bandit with μ₁ = 0.5
- uses Δ = 0.5, so μ₂ = 1.0
- sweeps the same learning rates as Part B
- runs `100` simulations per learning rate (overridable via `--n-simulations`)

It generates:

- `results/exp3_large_delta_lr_sweep.png`: regret-vs-learning-rate plot for Δ = 0.5
- `results/exp3_large_delta_lr_sweep.json`: empirical average regret and standard error for each learning rate
