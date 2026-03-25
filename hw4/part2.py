from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import BanditAlgorithm, ThompsonSampling, UpperConfidenceBound
from bandit_sim.bandits import GaussianBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42


@dataclass
class Part2Config:
    """Configuration for the experiment."""

    optimal_mean: float = 0.5
    suboptimal_tail_mean: float = -0.5
    reward_std: float = 1.0
    horizon: int = 2000
    n_simulations: int = 10
    deltas: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
            dtype=np.float64,
        )
    )
    seed: int = GLOBAL_SEED
    plot_output_path: Path = Path("results/part2_frequentist_regret.png")
    json_output_path: Path = Path("results/part2_frequentist_regret.json")


def empirical_regret_statistics(
    total_rewards: np.ndarray,
    optimal_expected_reward: float,
    horizon: int,
) -> tuple[float, float]:
    """Return empirical average regret and its standard error."""
    regrets = horizon * optimal_expected_reward - total_rewards
    average_regret = float(regrets.mean())
    if regrets.size <= 1:
        return average_regret, 0.0
    standard_error = float(regrets.std(ddof=1) / np.sqrt(regrets.size))
    return average_regret, standard_error


def build_bandit(delta: float, seed: int, config: Part2Config) -> GaussianBandit:
    """Create the fixed 10-armed Gaussian bandit for one delta value."""
    arm_means = (
        config.optimal_mean,  # optimal arm
        config.optimal_mean - float(delta),  # second-optimal arm
        *([config.suboptimal_tail_mean] * 8),  # suboptimal tail arms
    )
    arm_stds = tuple(float(config.reward_std) for _ in range(10))
    return GaussianBandit(arm_means=arm_means, arm_stds=arm_stds, seed=seed)


def build_algorithms() -> dict[str, BanditAlgorithm]:
    """Create fresh algorithm instances for one delta value."""
    return {
        "ucb": UpperConfidenceBound(seed=GLOBAL_SEED),
        "thompson_sampling": ThompsonSampling(
            seed=GLOBAL_SEED,
            prior_type="gaussian",
            prior_args=[{"mean": 0.0, "std": 1.0} for _ in range(10)],
        ),
    }


def run_part2_experiment(config: Part2Config) -> dict[str, dict[str, np.ndarray]]:
    """Run frequentist regret experiments across the requested deltas."""
    results: dict[str, dict[str, np.ndarray]] = {}

    for delta_index, delta in enumerate(config.deltas):
        bandit = build_bandit(delta=float(delta), seed=config.seed + delta_index, config=config)
        optimal_expected_reward = float(np.max(bandit.expected_rewards))

        print(f"delta={delta:.3f}")

        for name, algorithm in build_algorithms().items():
            if name not in results:
                results[name] = {
                    "average_regret": np.empty(config.deltas.shape[0], dtype=np.float64),
                    "standard_error": np.empty(config.deltas.shape[0], dtype=np.float64),
                }

            batch_result = algorithm.run_n_simulations(
                bandit=bandit,
                horizon=config.horizon,
                n_simulations=config.n_simulations,
            )
            average_regret, standard_error = empirical_regret_statistics(
                total_rewards=batch_result.total_rewards,
                optimal_expected_reward=optimal_expected_reward,
                horizon=config.horizon,
            )
            results[name]["average_regret"][delta_index] = average_regret
            results[name]["standard_error"][delta_index] = standard_error

            print(f"  {name}: average_regret={average_regret:.4f}, standard_error={standard_error:.4f}")

        print()

    return results


def build_json_summary(
    config: Part2Config,
    results: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the experiment."""
    bandit_problems: list[dict[str, object]] = []

    for delta_index, delta in enumerate(config.deltas):
        arm_means = [
            config.optimal_mean,
            config.optimal_mean - float(delta),
            *([config.suboptimal_tail_mean] * 8),
        ]
        algorithms: dict[str, dict[str, float]] = {}

        for name, metrics in results.items():
            algorithms[name] = {
                "empirical_average_regret": float(metrics["average_regret"][delta_index]),
                "standard_error": float(metrics["standard_error"][delta_index]),
            }

        bandit_problems.append(
            {
                "delta": float(delta),
                "arm_means": arm_means,
                "arm_stds": [config.reward_std] * 10,
                "algorithms": algorithms,
            }
        )

    return {
        "horizon": config.horizon,
        "n_simulations": config.n_simulations,
        "global_seed": config.seed,
        "bandit_problems": bandit_problems,
    }


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_regret_vs_delta(
    config: Part2Config,
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot frequentist regret against delta with error bars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    label_map = {
        "ucb": "UCB",
        "thompson_sampling": "TS",
    }
    colors = {
        "ucb": "tab:green",
        "thompson_sampling": "tab:blue",
    }

    for name, metrics in results.items():
        axis.errorbar(
            config.deltas,
            metrics["average_regret"],
            yerr=metrics["standard_error"],
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[name],
            label=label_map[name],
        )

    axis.set_xlabel(r"Gap $\Delta$")
    axis.set_ylabel(rf"Expected Regret at Horizon $n = {config.horizon}$")
    axis.set_title("Frequentist Regret")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    config = Part2Config()

    print("Part 2: Frequentist Regret")
    print(f"horizon={config.horizon}")
    print(f"n_simulations={config.n_simulations}")
    print(f"deltas={config.deltas.tolist()}")
    print()

    results = run_part2_experiment(config)
    plot_regret_vs_delta(config, results, config.plot_output_path)
    json_summary = build_json_summary(config, results)
    save_json_summary(json_summary, config.json_output_path)

    print(f"Saved plot to {config.plot_output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
