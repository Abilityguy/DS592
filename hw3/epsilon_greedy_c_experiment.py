"""Run a delta sweep for epsilon-greedy across multiple c values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import EpsilonGreedy
from bandit_sim.bandits import NArmedGaussianBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42


@dataclass
class EpsilonGreedyCSweepConfig:
    """Configuration for sweeping epsilon-greedy exploration constant c."""

    optimal_mean: float = 0
    optimal_std: float = 1.0
    suboptimal_std: float = 1.0
    deltas: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
            dtype=np.float64,
        )
    )
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0, 1000.0)
    horizon: int = 1000
    n_simulations: int = 100
    bandit_seed_base: int = GLOBAL_SEED
    output_path: Path = Path("results/epsilon_greedy_c_sweep.png")


def empirical_regret_statistics(
    total_rewards: np.ndarray,
    optimal_expected_reward: float,
    horizon: int,
) -> tuple[float, float]:
    """Return empirical average regret and standard error."""
    regrets = horizon * optimal_expected_reward - total_rewards
    average_regret = float(regrets.mean())
    if regrets.size <= 1:
        return average_regret, 0.0
    standard_error = float(regrets.std(ddof=1) / np.sqrt(regrets.size))
    return average_regret, standard_error


def run_c_sweep(config: EpsilonGreedyCSweepConfig) -> dict[str, dict[str, np.ndarray]]:
    """Run epsilon-greedy across c values for each delta."""
    results: dict[str, dict[str, np.ndarray]] = {}

    for c in config.c_values:
        key = f"c={c:g}"
        results[key] = {
            "average_regret": np.empty(config.deltas.shape[0], dtype=np.float64),
            "standard_error": np.empty(config.deltas.shape[0], dtype=np.float64),
        }

    for delta_index, delta in enumerate(config.deltas):
        bandit = NArmedGaussianBandit(
            arm_means=(config.optimal_mean, config.optimal_mean - float(delta)),
            arm_stds=(config.optimal_std, config.suboptimal_std),
            seed=config.bandit_seed_base + delta_index,
        )
        optimal_expected_reward = float(np.max(bandit.expected_rewards))
        print(f"delta={delta:.3f}")

        for c in config.c_values:
            key = f"c={c:g}"
            algorithm = EpsilonGreedy(c=c, seed=GLOBAL_SEED)
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
            results[key]["average_regret"][delta_index] = average_regret
            results[key]["standard_error"][delta_index] = standard_error
            print(f"  {key}: average_regret={average_regret:.4f}, standard_error={standard_error:.4f}")

        print()

    return results


def plot_c_sweep(
    config: EpsilonGreedyCSweepConfig,
    results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Plot empirical regret vs delta for each epsilon-greedy c value."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    color_map = {
        "c=0.1": "tab:orange",
        "c=1": "tab:green",
        "c=10": "tab:purple",
        "c=50": "tab:blue",
        "c=1000": "tab:red",
    }

    for label, metrics in results.items():
        axis.errorbar(
            config.deltas,
            metrics["average_regret"],
            yerr=metrics["standard_error"],
            marker="o",
            capsize=4,
            linewidth=2,
            color=color_map[label],
            label=label,
        )

    axis.set_xlabel(r"Gap $\Delta$")
    axis.set_ylabel(rf"Expected Regret at Horizon $n = {config.horizon}$")
    axis.set_title("Epsilon-Greedy: Regret vs Gap for Different c Values")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)

    figure.tight_layout()
    figure.savefig(config.output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run c-sweep experiment and save plot."""
    config = EpsilonGreedyCSweepConfig()

    print("Epsilon-Greedy c Sweep")
    print(f"c_values={list(config.c_values)}")
    print(f"horizon={config.horizon}, n_simulations={config.n_simulations}")
    print(f"deltas={config.deltas.tolist()}")
    print()

    results = run_c_sweep(config)
    plot_c_sweep(config, results)
    print(f"Saved plot to {config.output_path}")


if __name__ == "__main__":
    main()
