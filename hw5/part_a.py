"""Horizon sweep experiment for UCB and EXP3 on a 2-armed Bernoulli bandit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import EXP3, BanditAlgorithm, UpperConfidenceBound
from bandit_sim.bandits import BernoulliBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42


@dataclass
class HorizonSweepConfig:
    """Configuration for a two-armed regret-vs-horizon experiment."""

    mean_1: float = 0.5
    mean_2: float = 0.55  # Optimal arm
    horizons: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                10,
                100,
                1000,
                10000,
                100000,
            ],
            dtype=np.int_,
        )
    )
    n_simulations: int = 100
    master_seed: int = GLOBAL_SEED
    json_output_path: Path = Path("results/regret_vs_horizon.json")
    output_path: Path = Path("results/regret_vs_horizon.png")


def draw_seed(rng: np.random.Generator) -> int:
    """Draw an independent integer seed from a master RNG."""
    return int(rng.integers(0, 2**32 - 1))


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


def build_algorithms(
    horizon: int,
    n_arms: int,
    rng: np.random.Generator,
) -> dict[str, BanditAlgorithm]:
    """Create fresh algorithm instances."""
    # We set the learning rate for EXP3 according to a common theoretical choice
    exp3_lr = np.sqrt(2 * np.log(n_arms) / (horizon * n_arms))

    # Since we are using Bernoulli bandits, and the Bernoulli distribution is sub-gaussian with factor 1/4
    ucb_beta = 0.25
    return {
        "ucb": UpperConfidenceBound(seed=draw_seed(rng), beta=ucb_beta),
        "ucb_beta_1.0": UpperConfidenceBound(seed=draw_seed(rng), beta=1.0),
        "exp3": EXP3(seed=draw_seed(rng), learning_rate=exp3_lr),
    }


def run_horizon_sweep(config: HorizonSweepConfig) -> dict[str, dict[str, np.ndarray]]:
    """Run simulations across a range of horizon values."""
    results: dict[str, dict[str, np.ndarray]] = {}
    master_rng = np.random.default_rng(config.master_seed)

    for horizon_index, horizon in enumerate(config.horizons):
        bandit_seed = draw_seed(master_rng)
        optimal_expected_reward = float(max(config.mean_1, config.mean_2))

        print(f"horizon={horizon}")

        for name, algorithm in build_algorithms(horizon, 2, master_rng).items():
            if name not in results:
                results[name] = {
                    "average_regret": np.empty(config.horizons.shape[0], dtype=np.float64),
                    "standard_error": np.empty(config.horizons.shape[0], dtype=np.float64),
                }

            bandit = BernoulliBandit(
                arm_probs=(config.mean_1, config.mean_2),
                seed=bandit_seed,
            )
            batch_result = algorithm.run_n_simulations(
                bandit=bandit,
                horizon=int(horizon),
                n_simulations=config.n_simulations,
            )
            average_regret, standard_error = empirical_regret_statistics(
                total_rewards=batch_result.total_rewards,
                optimal_expected_reward=optimal_expected_reward,
                horizon=int(horizon),
            )
            results[name]["average_regret"][horizon_index] = average_regret
            results[name]["standard_error"][horizon_index] = standard_error

            print(f"  {name}: average_regret={average_regret:.4f}, standard_error={standard_error:.4f}")

        print()

    return results


def build_json_summary(
    config: HorizonSweepConfig,
    results: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the horizon sweep."""
    horizons_summary: list[dict[str, object]] = []

    for horizon_index, horizon in enumerate(config.horizons):
        algorithms: dict[str, dict[str, float]] = {}

        for name, metrics in results.items():
            algorithms[name] = {
                "empirical_average_regret": float(metrics["average_regret"][horizon_index]),
                "standard_error": float(metrics["standard_error"][horizon_index]),
            }

        horizons_summary.append({"horizon": int(horizon), "algorithms": algorithms})

    return {
        "mean_1": config.mean_1,
        "mean_2": config.mean_2,
        "n_simulations": config.n_simulations,
        "global_seed": config.master_seed,
        "horizons": horizons_summary,
    }


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_regret_vs_horizon(
    config: HorizonSweepConfig,
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot empirical average regret against horizon with error bars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {
        "ucb": "tab:blue",
        "ucb_beta_1.0": "tab:green",
        "exp3": "tab:orange",
    }
    label_map = {
        "ucb": r"UCB ($\beta=0.25$)",
        "ucb_beta_1.0": r"UCB ($\beta=1.0$)",
        "exp3": "EXP3",
    }

    for name, metrics in results.items():
        axis.errorbar(
            config.horizons,
            metrics["average_regret"],
            yerr=metrics["standard_error"],
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[name],
            label=label_map[name],
        )

    axis.set_xscale("log")
    axis.set_xlabel(r"Horizon $n$")
    axis.set_ylabel("Empirical Average Regret")
    axis.set_title(r"UCB vs EXP3 on a 2-Armed Bernoulli Bandit ($\mu_1=0.5$, $\mu_2=0.55$)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run the experiment and save the regret plot."""
    config = HorizonSweepConfig()

    print("2-Armed Bernoulli Bandit Horizon Sweep")
    print(f"mean_1={config.mean_1}, mean_2={config.mean_2}")
    print(f"horizons={config.horizons.tolist()}")
    print()

    results = run_horizon_sweep(config)
    plot_regret_vs_horizon(config, results, config.output_path)
    json_summary = build_json_summary(config, results)
    save_json_summary(json_summary, config.json_output_path)

    print(f"Saved plot to {config.output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
