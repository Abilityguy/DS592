"""Validate the variance behavior of loss-based EXP3 on an adversarial bandit."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GLOBAL_SEED = 42
N_ARMS = 2


@dataclass
class EXP3VarianceConfig:
    """Configuration for the adversarial EXP3 variance experiment."""

    horizon: int = 10000
    alphas: tuple[float, ...] = (0.28, 0.285, 0.29, 0.295, 0.30)
    n_simulations: int = 500
    master_seed: int = GLOBAL_SEED
    json_output_path: Path = Path("results/exp3_variance_boxplot.json")
    output_path: Path = Path("results/exp3_variance_boxplot.png")


def draw_seed(rng: np.random.Generator) -> int:
    """Draw an independent integer seed from a master RNG."""
    return int(rng.integers(0, 2**32 - 1))


def theoretical_learning_rate(horizon: int, n_arms: int) -> float:
    """Return the theoretical learning rate sqrt(2 ln K / (n K))."""
    return float(np.sqrt(2 * np.log(n_arms) / (horizon * n_arms)))


def adversarial_losses(step: int, horizon: int, alpha: float) -> np.ndarray:
    """Return the loss vector for the two-armed adversarial bandit."""
    if step < horizon // 2:
        return np.array([0.0, alpha], dtype=np.float64)
    return np.array([1.0, 0.0], dtype=np.float64)


def run_loss_exp3_simulation(
    alpha: float,
    horizon: int,
    learning_rate: float,
    seed: int,
) -> float:
    """Run one loss-based EXP3 simulation and return its regret. 
    The bandit_sim version cannot be used for an adversarial bandit case yet."""
    rng = np.random.default_rng(seed)
    estimated_loss_sums = np.zeros(N_ARMS, dtype=np.float64)
    learner_loss = 0.0

    for step in range(horizon):
        weights = np.exp(-learning_rate * estimated_loss_sums)
        probabilities = weights / weights.sum()

        action = int(rng.choice(N_ARMS, p=probabilities))
        losses = adversarial_losses(step, horizon, alpha)
        observed_loss = float(losses[action])
        learner_loss += observed_loss

        estimated_loss_sums[action] += observed_loss / probabilities[action]

    arm_1_loss = horizon / 2.0
    arm_2_loss = alpha * horizon / 2.0
    optimal_arm_loss = min(arm_1_loss, arm_2_loss)
    return float(learner_loss - optimal_arm_loss)


def run_variance_experiment(config: EXP3VarianceConfig) -> dict[float, np.ndarray]:
    """Run the EXP3 variance experiment across all alpha values."""
    if config.horizon % 2 != 0:
        raise ValueError("The experiment assumes an even horizon.")

    master_rng = np.random.default_rng(config.master_seed)
    learning_rate = theoretical_learning_rate(config.horizon, N_ARMS)
    regrets_by_alpha: dict[float, np.ndarray] = {}

    for alpha in config.alphas:
        regrets = np.empty(config.n_simulations, dtype=np.float64)
        for simulation_index in range(config.n_simulations):
            regrets[simulation_index] = run_loss_exp3_simulation(
                alpha=alpha,
                horizon=config.horizon,
                learning_rate=learning_rate,
                seed=draw_seed(master_rng),
            )

        regrets_by_alpha[alpha] = regrets
        print(
            f"alpha={alpha:.3f}: average_regret={regrets.mean():.4f}, "
            f"std={regrets.std(ddof=1) if regrets.size > 1 else 0.0:.4f}"
        )

    return regrets_by_alpha


def build_json_summary(
    config: EXP3VarianceConfig,
    regrets_by_alpha: dict[float, np.ndarray],
) -> dict[str, object]:
    """Build a JSON-serializable summary of the variance experiment."""
    learning_rate = theoretical_learning_rate(config.horizon, N_ARMS)
    return {
        "horizon": config.horizon,
        "n_arms": N_ARMS,
        "n_simulations": config.n_simulations,
        "global_seed": config.master_seed,
        "learning_rate": learning_rate,
        "alphas": [
            {
                "alpha": alpha,
                "average_regret": float(regrets.mean()),
                "standard_error": float(regrets.std(ddof=1) / np.sqrt(regrets.size))
                if regrets.size > 1
                else 0.0,
                "regrets": regrets.tolist(),
            }
            for alpha, regrets in regrets_by_alpha.items()
        ],
    }


def save_json_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write the experiment summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_regret_boxplot(
    config: EXP3VarianceConfig,
    regrets_by_alpha: dict[float, np.ndarray],
    output_path: Path,
) -> None:
    """Plot boxplots of regret across alpha values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    data = [regrets_by_alpha[alpha] for alpha in config.alphas]
    positions = np.arange(1, len(config.alphas) + 1)
    box_color = "#aeb7c2"

    boxplot = axis.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.6,
        boxprops={"edgecolor": "#4d4d4d"},
        medianprops={"color": "#1f1f1f", "linewidth": 1.5},
        whiskerprops={"color": "#4d4d4d"},
        capprops={"color": "#4d4d4d"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#4d4d4d",
            "markeredgecolor": "#4d4d4d",
            "markersize": 4,
            "alpha": 0.8,
        },
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_alpha(0.85)

    means = [float(np.mean(regrets_by_alpha[alpha])) for alpha in config.alphas]
    axis.scatter(positions, means, color="#2c2c2c", marker="D", s=25, zorder=3)

    axis.set_xticks(positions)
    axis.set_xticklabels([f"{alpha:.3f}".rstrip("0").rstrip(".") for alpha in config.alphas], rotation=45)
    axis.set_xlabel(r"$\alpha$")
    axis.set_ylabel("Regret")
    axis.set_title(r"Variance of EXP3 on the Adversarial Loss Sequence")
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run the EXP3 variance experiment and save outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-simulations", type=int, default=EXP3VarianceConfig.n_simulations)
    parser.add_argument("--horizon", type=int, default=EXP3VarianceConfig.horizon)
    args = parser.parse_args()

    config = EXP3VarianceConfig(n_simulations=args.n_simulations, horizon=args.horizon)
    learning_rate = theoretical_learning_rate(config.horizon, N_ARMS)

    print("EXP3 Variance Experiment")
    print(f"horizon={config.horizon}")
    print(f"n_simulations={config.n_simulations}")
    print(f"alphas={list(config.alphas)}")
    print(f"learning_rate={learning_rate:.6f}")
    print()

    regrets_by_alpha = run_variance_experiment(config)
    plot_regret_boxplot(config, regrets_by_alpha, config.output_path)
    json_summary = build_json_summary(config, regrets_by_alpha)
    save_json_summary(json_summary, config.json_output_path)

    print(f"Saved plot to {config.output_path}")
    print(f"Saved JSON summary to {config.json_output_path}")


if __name__ == "__main__":
    main()
