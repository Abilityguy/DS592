"""EXP3 learning rate sweep on a 2-armed Bernoulli bandit at a fixed horizon."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import EXP3
from bandit_sim.bandits import BernoulliBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42
N_ARMS = 2


@dataclass
class LearningRateSweepConfig:
    """Configuration for the EXP3 learning-rate sweep experiment."""

    mean_1: float = 0.5
    mean_2: float = 0.55  # Optimal arm
    horizon: int = 100000
    learning_rates: tuple[float, ...] = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
    n_simulations: int = 100
    master_seed: int = GLOBAL_SEED
    json_output_path: Path = Path("results/exp3_lr_sweep.json")
    output_path: Path = Path("results/exp3_lr_sweep.png")


def draw_seed(rng: np.random.Generator) -> int:
    """Draw an independent integer seed from a master RNG."""
    return int(rng.integers(0, 2**32 - 1))


def theoretical_learning_rate(horizon: int, n_arms: int) -> float:
    """Return the theory-optimal EXP3 learning rate sqrt(2 ln K / (T K))."""
    return float(np.sqrt(2 * np.log(n_arms) / (horizon * n_arms)))


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


def run_lr_sweep(config: LearningRateSweepConfig) -> dict[str, dict[str, float]]:
    """Run EXP3 simulations across learning rate values."""
    master_rng = np.random.default_rng(config.master_seed)
    theoretical_lr = theoretical_learning_rate(config.horizon, N_ARMS)
    all_lrs = list(config.learning_rates) + [theoretical_lr]
    bandit_seed = draw_seed(master_rng)
    optimal_expected_reward = float(max(config.mean_1, config.mean_2))

    results: dict[str, dict[str, float]] = {}

    for lr_index, lr in enumerate(all_lrs):
        key = f"theoretical ({lr:.5f})" if lr == theoretical_lr else f"{lr:g}"
        bandit = BernoulliBandit(
            arm_probs=(config.mean_1, config.mean_2),
            seed=bandit_seed,
        )
        algorithm = EXP3(seed=draw_seed(master_rng), learning_rate=lr)
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
        results[key] = {
            "learning_rate": lr,
            "average_regret": average_regret,
            "standard_error": standard_error,
        }
        print(f"  lr={key}: average_regret={average_regret:.4f}, standard_error={standard_error:.4f}")

    return results


def build_json_summary(
    config: LearningRateSweepConfig,
    results: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the learning rate sweep."""
    return {
        "mean_1": config.mean_1,
        "mean_2": config.mean_2,
        "horizon": config.horizon,
        "n_simulations": config.n_simulations,
        "global_seed": config.master_seed,
        "results": [
            {
                "label": key,
                "learning_rate": metrics["learning_rate"],
                "empirical_average_regret": metrics["average_regret"],
                "standard_error": metrics["standard_error"],
            }
            for key, metrics in results.items()
        ],
    }


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_regret_vs_lr(
    config: LearningRateSweepConfig,
    results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Plot empirical average regret against learning rate with error bars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))

    theoretical_lr = theoretical_learning_rate(config.horizon, N_ARMS)

    sorted_items = sorted(results.items(), key=lambda x: x[1]["learning_rate"])
    lrs = [m["learning_rate"] for _, m in sorted_items]
    regrets = [m["average_regret"] for _, m in sorted_items]
    errors = [m["standard_error"] for _, m in sorted_items]

    axis.errorbar(
        lrs,
        regrets,
        yerr=errors,
        marker="o",
        capsize=4,
        linewidth=2,
        color="tab:blue",
    )

    # Highlight the theoretical learning rate
    theo_metrics = next(m for _, m in sorted_items if m["learning_rate"] == theoretical_lr)
    axis.errorbar(
        theoretical_lr,
        theo_metrics["average_regret"],
        yerr=theo_metrics["standard_error"],
        marker="*",
        capsize=4,
        color="tab:red",
        markersize=12,
        label=r"theoretical $\eta$",
        zorder=5,
    )

    axis.set_xscale("log")
    axis.set_xlabel(r"Learning Rate $\eta$")
    axis.set_ylabel("Empirical Average Regret")
    axis.set_title(
        rf"EXP3 Learning Rate Sweep"
        rf" ($\mu_1=0.5$, $\mu_2=0.55$, $n={config.horizon}$)"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run the experiment and save the regret plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-simulations", type=int, default=LearningRateSweepConfig.n_simulations)
    args = parser.parse_args()

    config = LearningRateSweepConfig(n_simulations=args.n_simulations)
    theoretical_lr = theoretical_learning_rate(config.horizon, N_ARMS)

    print("EXP3 Learning Rate Sweep")
    print(f"mean_1={config.mean_1}, mean_2={config.mean_2}, horizon={config.horizon}")
    print(f"learning_rates={list(config.learning_rates)}")
    print(f"theoretical_lr={theoretical_lr:.6f}")
    print()

    results = run_lr_sweep(config)
    plot_regret_vs_lr(config, results, config.output_path)
    json_summary = build_json_summary(config, results)
    save_json_summary(json_summary, config.json_output_path)

    print(f"Saved plot to {config.output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
