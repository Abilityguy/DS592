"""Thompson Sampling algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any

import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms.base import BanditAlgorithm


@dataclass
class ThompsonSampling(BanditAlgorithm):
    """An implementation of the Thompson Sampling algorithm."""

    seed: int | None = None
    counts: npt.NDArray[np.int_] = field(init=False)
    prior_type: Literal["gaussian", "uniform"] = "gaussian"
    prior_args: list[dict] = field(default_factory=list)
    # Gaussian Prior variables
    _means: npt.NDArray[np.float64] = field(init=False)
    _stds: npt.NDArray[np.float64] = field(init=False)
    # Uniform Prior variables
    _mins: npt.NDArray[np.float64] = field(init=False)
    _maxs: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _initialize_state(self) -> None:
        assert self.n_arms is not None
        self.counts = np.zeros(self.n_arms, dtype=np.int_)

        if self.prior_type == "gaussian":
            self._means = np.array([x["mean"] for x in self.prior_args], dtype=np.float64)
            self._stds = np.array([x["std"] for x in self.prior_args], dtype=np.float64)
        elif self.prior_type == "uniform":
            self._mins = np.array([x["min"] for x in self.prior_args], dtype=np.float64)
            self._maxs = np.array([x["max"] for x in self.prior_args], dtype=np.float64)
        else:
            raise ValueError("Invalid prior_type. Must be 'gaussian' or 'uniform'.")

    def _prepare_run(self, horizon: int) -> None:
        self._posterior_args: list[Any] = [None for _ in range(self.n_arms)]
        self.arm_reward_history: list[list[float]] = [[] for _ in range(self.n_arms)]

    def select_arm(self) -> int:
        self._check_initialized()
        assert self.n_arms is not None

        # Sample mean from the posterior distribution for each arm
        if self.prior_type == "gaussian":
            sampled_means = [self._rng.normal(loc=loc, scale=scale) for loc, scale in zip(self._means, self._stds)]
        else:  # self.prior_type == "uniform"
            # For arms with no pulls, we assume uniform prior, but the rewards are actually Gaussian distributed, so we switch to Gaussian posterior after the first turn
            sampled_means = []
            for arm_idx, count in enumerate(self.counts):
                if count == 0:
                    sampled_means.append(self._rng.uniform(low=self._mins[arm_idx], high=self._maxs[arm_idx]))
                else:
                    sampled_mean = self._rng.normal(
                        loc=self._posterior_args[arm_idx]["loc"], scale=self._posterior_args[arm_idx]["scale"]
                    )
                    # We clip the sampled mean to be within the uniform prior bounds
                    truncated_sampled_mean = np.clip(sampled_mean, self._mins[arm_idx], self._maxs[arm_idx])
                    sampled_means.append(truncated_sampled_mean)

        # Select the arm with the highest sampled mean
        return int(np.argmax(sampled_means))

    def update(self, arm_index: int, reward: float) -> None:
        self._check_initialized()

        if arm_index < 0 or self.n_arms is None or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        self.total_steps += 1
        self.counts[arm_index] += 1
        self.arm_reward_history[arm_index].append(reward)

        if self.prior_type == "gaussian":
            # Update the Gaussian posterior parameters (mean and std)
            # In Thompson Sampling with Gaussian rewards and Gaussian prior, the posterior is also Gaussian
            rewards = np.array(self.arm_reward_history[arm_index], dtype=np.float64)
            loc = rewards.sum() / (self.counts[arm_index] + 1)
            scale = 1 / np.sqrt(self.counts[arm_index] + 1)
            self._posterior_args[arm_index] = {"loc": loc, "scale": scale}
        else:  # self.prior_type == "uniform"
            # We assume the rewards are Gaussian distributed, so the posterior is also a Gaussian
            rewards = np.array(self.arm_reward_history[arm_index], dtype=np.float64)
            loc = rewards.mean()
            scale = 1 / np.sqrt(self.counts[arm_index])
            self._posterior_args[arm_index] = {"loc": loc, "scale": scale}
