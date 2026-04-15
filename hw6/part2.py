"""Compare OFUL and UCB on the finite action set {-1, 1}^5."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from bandit_sim.algorithms import OFUL, UpperConfidenceBound
from bandit_sim.bandits import GaussianBandit, LinearSubGaussianBandit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_SEED = 42


@dataclass
class FiniteActionExperimentConfig:
    """Configuration for the {-1,1}^5 comparison experiment."""

    dimension: int = 5
    horizon: int = 10_000
    n_simulations: int = 10
    noise_scale: float = 1.0
    theta_star_upper_bound: float = 1.0
    master_seed: int = GLOBAL_SEED
    output_path: Path = Path("results/part2_hypercube_regret.png")
    json_output_path: Path = Path("results/part2_hypercube_regret.json")


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


def hypercube_actions(dimension: int) -> np.ndarray:
    """Return the 2^d action vectors in {-1, 1}^d."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=dimension)), dtype=np.float64)


def cumulative_pseudo_regret_for_oful(
    chosen_actions: np.ndarray,
    theta_star: np.ndarray,
    best_expected_reward: float,
) -> np.ndarray:
    """Compute cumulative pseudo-regret for OFUL from chosen actions."""
    chosen_expected_rewards = chosen_actions @ theta_star
    instant_regret = best_expected_reward - chosen_expected_rewards
    return np.cumsum(instant_regret)


def cumulative_pseudo_regret_for_ucb(
    chosen_arm_indices: np.ndarray,
    arm_means: np.ndarray,
    best_expected_reward: float,
) -> np.ndarray:
    """Compute cumulative pseudo-regret for UCB from chosen arm indices."""
    chosen_expected_rewards = arm_means[chosen_arm_indices]
    instant_regret = best_expected_reward - chosen_expected_rewards
    return np.cumsum(instant_regret)


def standard_error(curves: np.ndarray) -> np.ndarray:
    """Return the pointwise standard error across simulation curves."""
    if curves.shape[0] <= 1:
        return np.zeros(curves.shape[1], dtype=np.float64)
    return curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])


def run_finite_action_experiment(config: FiniteActionExperimentConfig) -> dict[str, object]:
    """Run OFUL and UCB on the finite action set {-1, 1}^5."""
    master_rng = np.random.default_rng(config.master_seed)
    oful_delta = 1.0 / config.horizon
    actions = hypercube_actions(config.dimension)
    theta_stars = np.empty((config.n_simulations, config.dimension), dtype=np.float64)
    oful_curves = np.empty((config.n_simulations, config.horizon), dtype=np.float64)
    ucb_curves = np.empty((config.n_simulations, config.horizon), dtype=np.float64)

    for simulation_index in range(config.n_simulations):
        theta_star = sample_unit_norm_theta(config.dimension, master_rng)
        theta_stars[simulation_index] = theta_star
        arm_means = actions @ theta_star
        best_expected_reward = float(np.max(arm_means))

        oful_bandit = LinearSubGaussianBandit(
            context_dimension_=config.dimension,
            theta_star=theta_star,
            context_sampler=lambda rng, action_matrix=actions: action_matrix,
            noise_scale=config.noise_scale,
            seed=draw_seed(master_rng),
        )
        oful = OFUL(
            delta=oful_delta,
            theta_star_upper_bound=config.theta_star_upper_bound,
            action_norm_bound=float(np.sqrt(config.dimension)),
        )
        oful_result = oful.run(bandit=oful_bandit, horizon=config.horizon)
        oful_curves[simulation_index] = cumulative_pseudo_regret_for_oful(
            chosen_actions=oful_result.chosen_actions,
            theta_star=theta_star,
            best_expected_reward=best_expected_reward,
        )

        ucb_bandit = GaussianBandit(
            arm_means=tuple(arm_means.tolist()),
            arm_stds=tuple(np.full(actions.shape[0], config.noise_scale, dtype=np.float64).tolist()),
            seed=draw_seed(master_rng),
        )
        ucb = UpperConfidenceBound(beta=config.noise_scale)
        ucb_result = ucb.run(bandit=ucb_bandit, horizon=config.horizon)
        ucb_curves[simulation_index] = cumulative_pseudo_regret_for_ucb(
            chosen_arm_indices=ucb_result.actions,
            arm_means=arm_means,
            best_expected_reward=best_expected_reward,
        )

    oful_final_regrets = oful_curves[:, -1]
    ucb_final_regrets = ucb_curves[:, -1]
    oful_final_regret_mean = float(oful_final_regrets.mean())
    ucb_final_regret_mean = float(ucb_final_regrets.mean())
    oful_final_regret_se = float(
        0.0 if oful_final_regrets.size <= 1 else oful_final_regrets.std(ddof=1) / np.sqrt(oful_final_regrets.size)
    )
    ucb_final_regret_se = float(
        0.0 if ucb_final_regrets.size <= 1 else ucb_final_regrets.std(ddof=1) / np.sqrt(ucb_final_regrets.size)
    )

    print(f"OFUL final_regret_mean={oful_final_regret_mean:.4f}, final_regret_se={oful_final_regret_se:.4f}")
    print(f"UCB final_regret_mean={ucb_final_regret_mean:.4f}, final_regret_se={ucb_final_regret_se:.4f}")

    return {
        "dimension": config.dimension,
        "actions": actions.tolist(),
        "theta_stars": theta_stars.tolist(),
        "oful": {
            "mean_cumulative_regret": oful_curves.mean(axis=0).tolist(),
            "cumulative_regret_standard_error": standard_error(oful_curves).tolist(),
            "final_regret_mean": oful_final_regret_mean,
            "final_regret_standard_error": oful_final_regret_se,
        },
        "ucb": {
            "mean_cumulative_regret": ucb_curves.mean(axis=0).tolist(),
            "cumulative_regret_standard_error": standard_error(ucb_curves).tolist(),
            "final_regret_mean": ucb_final_regret_mean,
            "final_regret_standard_error": ucb_final_regret_se,
        },
    }


def plot_finite_action_regret(
    config: FiniteActionExperimentConfig,
    results: dict[str, object],
    output_path: Path,
) -> None:
    """Plot mean cumulative pseudo-regret versus time with standard-error bands."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    time_axis = np.arange(1, config.horizon + 1)

    oful_mean = np.asarray(results["oful"]["mean_cumulative_regret"], dtype=np.float64)
    oful_se = np.asarray(results["oful"]["cumulative_regret_standard_error"], dtype=np.float64)
    axis.plot(
        time_axis,
        oful_mean,
        linewidth=2,
        color="tab:blue",
        label="OFUL",
    )
    axis.fill_between(
        time_axis,
        oful_mean - oful_se,
        oful_mean + oful_se,
        color="tab:blue",
        alpha=0.2,
    )

    ucb_mean = np.asarray(results["ucb"]["mean_cumulative_regret"], dtype=np.float64)
    ucb_se = np.asarray(results["ucb"]["cumulative_regret_standard_error"], dtype=np.float64)
    axis.plot(
        time_axis,
        ucb_mean,
        linewidth=2,
        color="tab:orange",
        label="UCB",
    )
    axis.fill_between(
        time_axis,
        ucb_mean - ucb_se,
        ucb_mean + ucb_se,
        color="tab:orange",
        alpha=0.2,
    )

    axis.set_xlabel(r"Time $t$")
    axis.set_ylabel("Mean Cumulative Pseudo-Regret")
    axis.set_title(r"OFUL vs UCB on $\{-1, 1\}^5$ with Standard-Error Bands")
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
    config: FiniteActionExperimentConfig,
    results: dict[str, object],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the finite-action experiment."""
    return {
        "dimension": config.dimension,
        "horizon": config.horizon,
        "delta": 1.0 / config.horizon,
        "n_simulations": config.n_simulations,
        "noise_scale": config.noise_scale,
        "theta_star_upper_bound": config.theta_star_upper_bound,
        "master_seed": config.master_seed,
        **results,
    }


def main() -> None:
    """Run the finite-action comparison experiment and save the outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=FiniteActionExperimentConfig.horizon)
    parser.add_argument("--n-simulations", type=int, default=FiniteActionExperimentConfig.n_simulations)
    args = parser.parse_args()

    config = FiniteActionExperimentConfig(horizon=args.horizon, n_simulations=args.n_simulations)

    print(r"HW6 Part 2: OFUL vs UCB on {-1, 1}^5")
    print(
        f"dimension={config.dimension}, horizon={config.horizon}, "
        f"n_simulations={config.n_simulations}, delta={1.0 / config.horizon}"
    )
    print()

    results = run_finite_action_experiment(config)
    plot_finite_action_regret(config, results, config.output_path)
    save_json_summary(build_json_summary(config, results), config.json_output_path)

    print(f"Saved plot to {config.output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
