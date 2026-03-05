"""Epsilon-greedy algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms.base import BanditAlgorithm


@dataclass
class EpsilonGreedy(BanditAlgorithm):
    """An implementation of the epsilon-greedy algorithm."""

    seed: int | None = None
    counts: npt.NDArray[np.int_] = field(init=False)
    value_estimates: npt.NDArray[np.float64] = field(init=False)
    c: float = field(default=50.0)  # Exploration parameter

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _initialize_state(self) -> None:
        assert self.n_arms is not None
        self.counts = np.zeros(self.n_arms, dtype=np.int_)
        self.value_estimates = np.zeros(self.n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        self._check_initialized()
        assert self.n_arms is not None

        # Explore: select a random arm
        epsilon = min(1.0, self.c / max(1, self.total_steps))
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_arms))

        # Exploit: select the arm with the highest estimated value
        return int(np.argmax(self.value_estimates))

    def update(self, arm_index: int, reward: float) -> None:
        self._check_initialized()

        if arm_index < 0 or self.n_arms is None or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        self.total_steps += 1
        self.counts[arm_index] += 1

        count = self.counts[arm_index]
        estimate = self.value_estimates[arm_index]
        self.value_estimates[arm_index] = estimate + (reward - estimate) / count
