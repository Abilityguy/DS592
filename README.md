# Bandit Sim

A library for running bandit experiments.

## Layout

```text
src/bandit_sim/
hw1/
hw3/
hw4/
hw5/
```

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install developer tooling:

```bash
pip install -e ".[dev]"
```

Install homework experiment dependencies:

```bash
pip install -r requirements.txt
```

## Quick Example

```python
from bandit_sim.algorithms import EpsilonGreedy
from bandit_sim.bandits import GaussianBandit

bandit = GaussianBandit(
    arm_means=(0.3, 0.7, 0.5),
    arm_stds=(1.0, 1.0, 1.0),
    seed=7,
)
algorithm = EpsilonGreedy(c=50.0, seed=7)

result = algorithm.run(bandit=bandit, horizon=100)

print(result.total_reward)
print(result.actions[:10])
```

## Multiple Runs

```python
batch = algorithm.run_n_simulations(
    bandit=bandit,
    horizon=100,
    n_simulations=10,
)

print(batch.total_rewards)
print(batch.average_total_reward)
```

## Formatting

Format the code with:

```bash
ruff format .
```
