# HW6 Experiments

This directory contains the Homework 6 experiment scripts, plots, and results. It expects the shared `bandit_sim` library from the repository root to already be installed.

## Structure

```text
part1.py
part2.py
```

## Design

- The shared library code is maintained outside `hw6` in `../src/bandit_sim`.
- `part1.py` runs OFUL on the unit-ball linear bandit action set for `d=5` and `d=10`.
- `part2.py` compares OFUL and multi-armed-bandit UCB on the finite action set `{-1, 1}^5`.
- In each script, `theta_star` is sampled randomly from a standard Gaussian direction and normalized to have Euclidean norm `1`.
- Both scripts set the OFUL confidence level to $\delta = 1/n$, where $n$ is the experiment horizon.

- Environment setup is documented in the repository root `README.md`.

## Part 1: Unit-Ball Action Set

Run Part 1 with:

```bash
python part1.py [--horizon T] [--n-simulations N]
```

This experiment:

- uses the action set $A_t = B_d(0,1)$
- runs OFUL for `d=5` and `d=10`
- uses default horizon `10000` (overridable with `--horizon`)
- runs `10` simulations per dimension by default (overridable with `--n-simulations`)
- samples a fresh `theta_star` in each simulation
- plots mean cumulative pseudo-regret versus time with standard-error bands

It generates:

- `results/part1_unit_ball_regret.png`: mean cumulative pseudo-regret versus time for `d=5` and `d=10`, with standard-error bands
- `results/part1_unit_ball_regret.json`: sampled `theta_star` values, mean cumulative regret curves, standard errors, and final regret summaries

> **Note:** Part 1 can take a while because OFUL solves a constrained optimization problem at every step for the unit-ball action set.

## Part 2: Finite Action Set `{-1, 1}^5`

Run Part 2 with:

```bash
python part2.py [--horizon T] [--n-simulations N]
```

This experiment:

- uses the finite action set $\{-1,1\}^5$, which has `32` actions
- runs OFUL on the linear bandit formulation
- runs UCB on the corresponding `32`-armed Gaussian bandit with arm means $a^\top \theta_\star$
- uses default horizon `10000` (overridable with `--horizon`)
- runs `10` simulations by default (overridable with `--n-simulations`)
- samples a fresh `theta_star` in each simulation and evaluates both algorithms on that same sampled problem
- plots mean cumulative pseudo-regret versus time for both algorithms with standard-error bands

It generates:

- `results/part2_hypercube_regret.png`: mean cumulative pseudo-regret of OFUL and UCB versus time, with standard-error bands
- `results/part2_hypercube_regret.json`: sampled `theta_star` values, the action set, mean cumulative regret curves, standard errors, and final regret summaries
