# HW3 Experiments

This directory contains the Homework 3 experiment scripts, plots, and results.

## Structure

```text
main.py
epsilon_greedy_c_experiment.py
assets/
results/
```

## Design

- The shared library code is maintained outside `hw3` in `../src/bandit_sim`.
- `main.py` runs the main two-armed Gaussian bandit delta sweep and saves both a plot and a JSON summary.
- `epsilon_greedy_c_experiment.py` runs the same delta sweep but only for epsilon-greedy across multiple `c` values.
- Environment setup is documented in the repository root `README.md`.

## Delta Sweep Experiment

Run the main experiment with:

```bash
python main.py
```

This generates:

- `results/regret_vs_delta.png`: empirical average regret vs. delta with error bars
- `results/regret_vs_delta.json`: bandit means plus regret and standard error for each algorithm

The current experiment compares:

- `EpsilonGreedy (c=10)`
- `EpsilonGreedy (c=50)`
- `ExploreThenCommit`
- `SuccessiveElimination`

on a two-armed Gaussian bandit where one arm is optimal and the other is separated by a gap `delta`.

The plot also overlays theoretical bounds for:

- `EpsilonGreedy (c=50)`
- `ExploreThenCommit`
- `SuccessiveElimination`

### Plot

![Regret vs Delta](assets/regret_vs_delta.png)

## Epsilon-Greedy c Sweep

Run the epsilon-greedy-only experiment with:

```bash
python epsilon_greedy_c_experiment.py
```

This evaluates epsilon-greedy for:

- `c = 0.1`
- `c = 1`
- `c = 10`
- `c = 50`
- `c = 1000`

and saves the plot to `results/epsilon_greedy_c_sweep.png`.

### Plot

![Epsilon-Greedy c Sweep](assets/epsilon_greedy_c_sweep.png)

## Outputs

The JSON summary stores, for each delta:

- arm means and standard deviations
- empirical average regret for each algorithm
- standard error for each algorithm

## Formatting

Format the code with:

```bash
ruff format .
```
