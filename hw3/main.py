"""Run a delta sweep experiment for a 2-armed Gaussian bandit."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import (
    BanditAlgorithm,
    EpsilonGreedy,
    ExploreThenCommit,
    SuccessiveElimination,
)
from bandit_sim.bandits import NArmedGaussianBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42


@dataclass
class DeltaSweepConfig:
    """Configuration for a two-armed regret-vs-delta experiment."""

    optimal_mean: float = 0
    optimal_std: float = 1.0
    suboptimal_std: float = 1.0
    deltas: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=np.float64
        )
    )
    horizon: int = 1000
    n_simulations: int = 100
    bandit_seed_base: int = GLOBAL_SEED
    json_output_path: Path = Path("results/regret_vs_delta.json")
    output_path: Path = Path("results/regret_vs_delta.png")


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


def build_algorithms() -> dict[str, BanditAlgorithm]:
    """Create fresh algorithm instances for one bandit instance."""
    return {
        "epsilon_greedy": EpsilonGreedy(c=50.0, seed=GLOBAL_SEED),
        "explore_then_commit": ExploreThenCommit(m=None, seed=GLOBAL_SEED),
        "successive_elimination": SuccessiveElimination(seed=GLOBAL_SEED),
    }


def run_delta_sweep(config: DeltaSweepConfig) -> dict[str, dict[str, np.ndarray]]:
    """Run simulations across a range of delta values."""
    results: dict[str, dict[str, np.ndarray]] = {}

    for delta_index, delta in enumerate(config.deltas):
        bandit = NArmedGaussianBandit(
            arm_means=(config.optimal_mean, config.optimal_mean - float(delta)),
            arm_stds=(config.optimal_std, config.suboptimal_std),
            seed=config.bandit_seed_base + delta_index,
        )
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

            print(
                f"  {name}: average_regret={average_regret:.4f}, "
                f"standard_error={standard_error:.4f}"
            )

        print()

    return results


def build_json_summary(
    config: DeltaSweepConfig,
    results: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the delta sweep."""
    bandit_problems: list[dict[str, object]] = []

    for delta_index, delta in enumerate(config.deltas):
        arm_means = [config.optimal_mean, config.optimal_mean - float(delta)]
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
                "arm_stds": [config.optimal_std, config.suboptimal_std],
                "algorithms": algorithms,
            }
        )

    return {
        "optimal_mean": config.optimal_mean,
        "optimal_std": config.optimal_std,
        "suboptimal_std": config.suboptimal_std,
        "horizon": config.horizon,
        "n_simulations": config.n_simulations,
        "global_seed": GLOBAL_SEED,
        "bandit_problems": bandit_problems,
    }


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_regret_vs_delta(
    config: DeltaSweepConfig,
    deltas: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot empirical average regret against delta with error bars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {
        "epsilon_greedy": "tab:blue",
        "explore_then_commit": "tab:orange",
        "successive_elimination": "tab:green",
    }

    for name, metrics in results.items():
        axis.errorbar(
            deltas,
            metrics["average_regret"],
            yerr=metrics["standard_error"],
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[name],
            label=name,
        )

    # Plot theoretical upper bounds on regret

    # ETC bound
    etc_m = int(np.ceil(config.horizon ** (2.0 / 3.0)))
    etc_bound = deltas * etc_m + deltas * (config.horizon - etc_m) * np.exp(-etc_m * deltas**2 / 4)

    # Successive Elimination bound
    n_arms = 2
    se_bound = np.full_like(deltas, np.sqrt(n_arms * config.horizon * np.log(config.horizon)))

    # Epsilon-Greedy bound
    epsilon_greedy_c = 50.0
    eg_bound = epsilon_greedy_c * deltas + deltas * config.horizon / epsilon_greedy_c

    axis.plot(
        deltas,
        eg_bound,
        "--",
        color=colors["epsilon_greedy"],
        linewidth=2,
        alpha=0.8,
        label="epsilon_greedy_bound",
    )
    axis.plot(
        deltas,
        etc_bound,
        "--",
        color=colors["explore_then_commit"],
        linewidth=2,
        alpha=0.8,
        label="explore_then_commit_bound",
    )
    axis.plot(
        deltas,
        se_bound,
        "--",
        color=colors["successive_elimination"],
        linewidth=2,
        alpha=0.8,
        label="successive_elimination_bound",
    )

    axis.set_xlabel(r"Gap $\Delta$")
    axis.set_ylabel(rf"Expected Regret at Horizon $n = {config.horizon}$")
    axis.set_title("Monte Carlo Simulations on a 2-Armed Bandit Problem")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", fontsize=7)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run the experiment and save the regret plot."""
    config = DeltaSweepConfig()

    print("2-Armed Gaussian Bandit Delta Sweep")
    print(f"optimal_mean={config.optimal_mean}, horizon={config.horizon}")
    print(f"n_simulations={config.n_simulations}")
    print(f"deltas={config.deltas.tolist()}")
    print()

    results = run_delta_sweep(config)
    plot_regret_vs_delta(config, config.deltas, results, config.output_path)
    json_summary = build_json_summary(config, results)
    save_json_summary(json_summary, config.json_output_path)

    print(f"Saved plot to {config.output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
