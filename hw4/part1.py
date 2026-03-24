from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms import ThompsonSampling
from bandit_sim.bandits import GaussianBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GLOBAL_SEED = 42


@dataclass
class Part1Config:
    """Configuration for the experiment."""

    n_arms: int = 10
    reward_std: float = 1.0
    prior_mean: float = 0.0
    prior_std: float = 1.0
    horizons: tuple[int, ...] = (100, 1000, 10000)
    n_simulations: int = 50
    seed: int = GLOBAL_SEED
    plot_output_path: Path = Path("results/part1_bayes_regret.png")
    json_output_path: Path = Path("results/part1_bayes_regret.json")


def empirical_bayes_regret(
    actions: npt.NDArray[np.int_],
    arm_means: npt.NDArray[np.float64],
) -> float:
    """Compute regret from the chosen actions and the true arm means."""
    optimal_mean = float(np.max(arm_means))
    chosen_means = arm_means[actions]
    return float(np.sum(optimal_mean - chosen_means))


def standard_error(samples: npt.NDArray[np.float64]) -> float:
    """Compute the standard error of a sample mean."""
    if samples.size <= 1:
        return 0.0
    return float(samples.std(ddof=1) / np.sqrt(samples.size))


def build_well_specified_ts(config: Part1Config, seed: int) -> ThompsonSampling:
    """Create Thompson Sampling with the correct Gaussian prior."""
    return ThompsonSampling(
        seed=seed,
        prior_type="gaussian",
        prior_args=[{"mean": config.prior_mean, "std": config.prior_std} for _ in range(config.n_arms)],
    )


def build_misspecified_ts(config: Part1Config, seed: int) -> ThompsonSampling:
    """Create Thompson Sampling with the misspecified uniform prior."""
    return ThompsonSampling(
        seed=seed,
        prior_type="uniform",
        prior_args=[{"min": -1.0, "max": 1.0} for _ in range(config.n_arms)],
    )


def run_single_simulation(
    algorithm: ThompsonSampling,
    arm_means: npt.NDArray[np.float64],
    reward_std: float,
    horizon: int,
    bandit_seed: int,
) -> float:
    """Run one simulation and return its Bayesian regret."""
    bandit = GaussianBandit(
        arm_means=tuple(float(x) for x in arm_means),
        arm_stds=tuple(float(reward_std) for _ in range(arm_means.shape[0])),
        seed=bandit_seed,
    )
    result = algorithm.run(bandit=bandit, horizon=horizon)
    return empirical_bayes_regret(actions=result.actions, arm_means=arm_means)


def run_part1_experiment(
    config: Part1Config,
) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
    """Run Bayesian regret experiments across the requested horizons."""
    rng = np.random.default_rng(config.seed)
    algorithm_names = ("well_specified", "misspecified_uniform")
    results = {
        name: {
            "average_regret": np.empty(len(config.horizons), dtype=np.float64),
            "standard_error": np.empty(len(config.horizons), dtype=np.float64),
        }
        for name in algorithm_names
    }

    for horizon_index, horizon in enumerate(config.horizons):
        regrets_by_algorithm = {name: np.empty(config.n_simulations, dtype=np.float64) for name in algorithm_names}

        print(f"horizon={horizon}")

        for simulation_index in range(config.n_simulations):
            arm_means = rng.normal(
                loc=config.prior_mean,
                scale=config.prior_std,
                size=config.n_arms,
            ).astype(np.float64)

            well_specified_algorithm = build_well_specified_ts(
                config=config,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            misspecified_algorithm = build_misspecified_ts(
                config=config,
                seed=int(rng.integers(0, 2**32 - 1)),
            )

            regrets_by_algorithm["well_specified"][simulation_index] = run_single_simulation(
                algorithm=well_specified_algorithm,
                arm_means=arm_means,
                reward_std=config.reward_std,
                horizon=horizon,
                bandit_seed=int(rng.integers(0, 2**32 - 1)),
            )
            regrets_by_algorithm["misspecified_uniform"][simulation_index] = run_single_simulation(
                algorithm=misspecified_algorithm,
                arm_means=arm_means,
                reward_std=config.reward_std,
                horizon=horizon,
                bandit_seed=int(rng.integers(0, 2**32 - 1)),
            )

        for name in algorithm_names:
            regrets = regrets_by_algorithm[name]
            average_regret = float(regrets.mean())
            error = standard_error(regrets)
            results[name]["average_regret"][horizon_index] = average_regret
            results[name]["standard_error"][horizon_index] = error
            print(f"  {name}: average_regret={average_regret:.4f}, standard_error={error:.4f}")

        print()

    return results


def build_json_summary(
    config: Part1Config,
    results: dict[str, dict[str, npt.NDArray[np.float64]]],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the experiment."""
    horizons_summary: list[dict[str, object]] = []

    for horizon_index, horizon in enumerate(config.horizons):
        horizons_summary.append(
            {
                "horizon": horizon,
                "algorithms": {
                    name: {
                        "empirical_average_bayes_regret": float(metrics["average_regret"][horizon_index]),
                        "standard_error": float(metrics["standard_error"][horizon_index]),
                    }
                    for name, metrics in results.items()
                },
            }
        )

    return {
        "n_arms": config.n_arms,
        "reward_std": config.reward_std,
        "prior_mean": config.prior_mean,
        "prior_std": config.prior_std,
        "horizons": list(config.horizons),
        "n_simulations": config.n_simulations,
        "global_seed": config.seed,
        "results_by_horizon": horizons_summary,
    }


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_bayes_regret(
    config: Part1Config,
    results: dict[str, dict[str, npt.NDArray[np.float64]]],
    output_path: Path,
) -> None:
    """Plot Bayes regret against the time horizon."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    horizons = np.asarray(config.horizons, dtype=np.int_)
    label_map = {
        "well_specified": r"TS with true prior $P$",
        "misspecified_uniform": r"TS with misspecified prior $P'$",
    }
    colors = {
        "well_specified": "tab:blue",
        "misspecified_uniform": "tab:orange",
    }

    for name, metrics in results.items():
        axis.errorbar(
            horizons,
            metrics["average_regret"],
            yerr=metrics["standard_error"],
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[name],
            label=label_map[name],
        )

    axis.set_xscale("log")
    axis.set_xticks(horizons)
    axis.set_xticklabels([str(horizon) for horizon in horizons])
    axis.set_xlabel(r"Horizon $n$")
    axis.set_ylabel("Bayes Regret")
    axis.set_title("Thompson Sampling Bayesian Regret")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    config = Part1Config()

    print("Part 1: Bayesian Regret")
    print(f"n_arms={config.n_arms}")
    print(f"horizons={list(config.horizons)}")
    print(f"n_simulations={config.n_simulations}")
    print()

    results = run_part1_experiment(config)
    plot_bayes_regret(config, results, config.plot_output_path)
    json_summary = build_json_summary(config, results)
    save_json_summary(json_summary, config.json_output_path)

    print(f"Saved plot to {config.plot_output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
