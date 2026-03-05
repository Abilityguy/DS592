"""Explore-Then-Commit algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms.base import BanditAlgorithm


@dataclass
class ExploreThenCommit(BanditAlgorithm):
    """An implementation of the Explore-Then-Commit algorithm."""

    m: int | None = (
        None  # Number of exploration rounds, if None, will be set to ceil(horizon^(2/3)) at the start of each run.
    )
    seed: int | None = None
    counts: npt.NDArray[np.int_] = field(init=False)
    value_estimates: npt.NDArray[np.float64] = field(init=False)
    commit_arm: int | None = field(default=None, init=False)
    _active_exploration_rounds: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.m is not None and self.m <= 0:
            raise ValueError("m must be a positive integer.")
        self._rng = np.random.default_rng(self.seed)

    def _initialize_state(self) -> None:
        assert self.n_arms is not None
        self.counts = np.zeros(self.n_arms, dtype=np.int_)
        self.value_estimates = np.zeros(self.n_arms, dtype=np.float64)
        self.commit_arm = None

    def _prepare_run(self, horizon: int) -> None:
        if self.m is None:
            self._active_exploration_rounds = int(np.ceil(horizon ** (2.0 / 3.0)))
        else:
            self._active_exploration_rounds = self.m

    def select_arm(self) -> int:
        self._check_initialized()
        assert self.n_arms is not None

        if self.total_steps < self._active_exploration_rounds:
            return int(self.total_steps % self.n_arms)  # Round-robin exploration of arms

        if self.commit_arm is None:
            self.commit_arm = int(np.argmax(self.value_estimates))
        return self.commit_arm

    def update(self, arm_index: int, reward: float) -> None:
        self._check_initialized()

        if arm_index < 0 or self.n_arms is None or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        self.total_steps += 1
        self.counts[arm_index] += 1

        count = int(self.counts[arm_index])
        estimate = self.value_estimates[arm_index]
        self.value_estimates[arm_index] = estimate + (reward - estimate) / count
