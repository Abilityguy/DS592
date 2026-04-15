"""OFUL on unit-ball linear bandits for d=5 and d=10."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import OFUL
from bandit_sim.bandits import LinearSubGaussianBandit, UnitBallActionSet

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42


@dataclass
class UnitBallExperimentConfig:
    """Configuration for the unit-ball OFUL experiment."""

    dimensions: tuple[int, ...] = (5, 10)
    horizon: int = 10_000
    n_simulations: int = 10
    noise_scale: float = 1.0
    theta_star_upper_bound: float = 1.0
    master_seed: int = GLOBAL_SEED
    output_path: Path = Path("results/part1_unit_ball_regret.png")
    json_output_path: Path = Path("results/part1_unit_ball_regret.json")


def draw_seed(rng: np.random.Generator) -> int:
    """Draw an independent integer seed from a master RNG."""
    return int(rng.integers(0, 2**32 - 1))


def sample_unit_norm_theta(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random parameter vector with Euclidean norm 1."""
    theta = rng.normal(size=dimension)
    theta_norm = float(np.linalg.norm(theta, ord=2))
    if theta_norm == 0.0:
        theta[0] = 1.0
        theta_norm = 1.0
    return theta / theta_norm


def cumulative_pseudo_regret(
    chosen_actions: np.ndarray,
    theta_star: np.ndarray,
    best_expected_reward: float,
) -> np.ndarray:
    """Compute cumulative pseudo-regret from chosen actions."""
    chosen_expected_rewards = chosen_actions @ theta_star
    instant_regret = best_expected_reward - chosen_expected_rewards
    return np.cumsum(instant_regret)


def standard_error(curves: np.ndarray) -> np.ndarray:
    """Return the pointwise standard error across simulation curves."""
    if curves.shape[0] <= 1:
        return np.zeros(curves.shape[1], dtype=np.float64)
    return curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])


def run_unit_ball_experiment(config: UnitBallExperimentConfig) -> dict[str, dict[str, object]]:
    """Run OFUL on unit-ball linear bandits for each requested dimension."""
    master_rng = np.random.default_rng(config.master_seed)
    oful_delta = 1.0 / config.horizon
    results: dict[str, dict[str, object]] = {}

    for dimension in config.dimensions:
        cumulative_regret_curves = np.empty((config.n_simulations, config.horizon), dtype=np.float64)
        theta_stars = np.empty((config.n_simulations, dimension), dtype=np.float64)
        final_regrets = np.empty(config.n_simulations, dtype=np.float64)

        for simulation_index in range(config.n_simulations):
            theta_star = sample_unit_norm_theta(dimension, master_rng)
            theta_stars[simulation_index] = theta_star
            best_expected_reward = float(np.linalg.norm(theta_star, ord=2))

            bandit = LinearSubGaussianBandit(
                context_dimension_=dimension,
                theta_star=theta_star,
                context_sampler=lambda rng, d=dimension: UnitBallActionSet(dimension_=d, radius=1.0),
                noise_scale=config.noise_scale,
                seed=draw_seed(master_rng),
            )
            algorithm = OFUL(
                delta=oful_delta,
                theta_star_upper_bound=config.theta_star_upper_bound,
                action_norm_bound=1.0,
            )
            simulation_result = algorithm.run(bandit=bandit, horizon=config.horizon)
            cumulative_regret = cumulative_pseudo_regret(
                chosen_actions=simulation_result.chosen_actions,
                theta_star=theta_star,
                best_expected_reward=best_expected_reward,
            )
            cumulative_regret_curves[simulation_index] = cumulative_regret
            final_regrets[simulation_index] = cumulative_regret[-1]

        mean_cumulative_regret = cumulative_regret_curves.mean(axis=0)
        cumulative_regret_se = standard_error(cumulative_regret_curves)
        final_regret_mean = float(final_regrets.mean())
        final_regret_se = float(0.0 if final_regrets.size <= 1 else final_regrets.std(ddof=1) / np.sqrt(final_regrets.size))

        results[str(dimension)] = {
            "dimension": dimension,
            "theta_stars": theta_stars.tolist(),
            "mean_cumulative_regret": mean_cumulative_regret.tolist(),
            "cumulative_regret_standard_error": cumulative_regret_se.tolist(),
            "final_regret_mean": final_regret_mean,
            "final_regret_standard_error": final_regret_se,
        }
        print(
            f"d={dimension}: final_regret_mean={final_regret_mean:.4f}, "
            f"final_regret_se={final_regret_se:.4f}"
        )

    return results


def plot_unit_ball_regret(
    config: UnitBallExperimentConfig,
    results: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    """Plot mean cumulative pseudo-regret versus time with standard-error bands."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    time_axis = np.arange(1, config.horizon + 1)
    colors = {5: "tab:blue", 10: "tab:orange"}

    for dimension_key, summary in results.items():
        dimension = int(dimension_key)
        cumulative_regret = np.asarray(summary["mean_cumulative_regret"], dtype=np.float64)
        cumulative_regret_se = np.asarray(
            summary["cumulative_regret_standard_error"],
            dtype=np.float64,
        )
        color = colors.get(dimension, None)
        axis.plot(
            time_axis,
            cumulative_regret,
            linewidth=2,
            color=color,
            label=rf"OFUL, $d={dimension}$",
        )
        axis.fill_between(
            time_axis,
            cumulative_regret - cumulative_regret_se,
            cumulative_regret + cumulative_regret_se,
            color=color,
            alpha=0.2,
        )

    axis.set_xlabel(r"Time $t$")
    axis.set_ylabel("Mean Cumulative Pseudo-Regret")
    axis.set_title(r"OFUL on Unit-Ball Linear Bandits with Standard-Error Bands")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_json_summary(
    config: UnitBallExperimentConfig,
    results: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the unit-ball experiment."""
    return {
        "dimensions": list(config.dimensions),
        "horizon": config.horizon,
        "delta": 1.0 / config.horizon,
        "n_simulations": config.n_simulations,
        "noise_scale": config.noise_scale,
        "theta_star_upper_bound": config.theta_star_upper_bound,
        "master_seed": config.master_seed,
        "results": [results[str(dimension)] for dimension in config.dimensions],
    }


def main() -> None:
    """Run the unit-ball OFUL experiment and save the outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=UnitBallExperimentConfig.horizon)
    parser.add_argument("--n-simulations", type=int, default=UnitBallExperimentConfig.n_simulations)
    args = parser.parse_args()

    config = UnitBallExperimentConfig(horizon=args.horizon, n_simulations=args.n_simulations)

    print("HW6 Part 1: OFUL on Unit-Ball Linear Bandits")
    print(
        f"dimensions={list(config.dimensions)}, horizon={config.horizon}, "
        f"n_simulations={config.n_simulations}, delta={1.0 / config.horizon}"
    )
    print()

    results = run_unit_ball_experiment(config)
    plot_unit_ball_regret(config, results, config.output_path)
    save_json_summary(build_json_summary(config, results), config.json_output_path)

    print(f"Saved plot to {config.output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
