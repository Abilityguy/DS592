"""Explore-Then-Commit algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms.base import BanditAlgorithm


@dataclass
class SuccessiveElimination(BanditAlgorithm):
    """An implementation of the Successive Elimination algorithm."""

    seed: int | None = None
    counts: npt.NDArray[np.int_] = field(init=False)
    value_estimates: npt.NDArray[np.float64] = field(init=False)
    active_arms: npt.NDArray[np.bool_] = field(init=False)
    _run_horizon: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _initialize_state(self) -> None:
        assert self.n_arms is not None
        self.counts = np.zeros(self.n_arms, dtype=np.int_)
        self.value_estimates = np.zeros(self.n_arms, dtype=np.float64)
        self.active_arms = np.ones(self.n_arms, dtype=np.bool_)

    def _prepare_run(self, horizon: int) -> None:
        self._run_horizon = horizon

    def select_arm(self) -> int:
        self._check_initialized()
        assert self.n_arms is not None

        # We play each active arm once
        active_arms = np.where(self.active_arms)[0]
        active_arm_counts = self.counts[active_arms]
        max_count = np.max(
            active_arm_counts
        )  # Should be one greater than the minimum count among active arms
        arms_to_play = active_arms[active_arm_counts < max_count]
        if len(arms_to_play) > 0:
            return int(self._rng.choice(arms_to_play))
        else:  # Start of the round when all active arms have been played the same number of times
            return int(self._rng.choice(active_arms))

    def update(self, arm_index: int, reward: float) -> None:
        self._check_initialized()

        if arm_index < 0 or self.n_arms is None or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        self.total_steps += 1
        self.counts[arm_index] += 1

        count = int(self.counts[arm_index])
        estimate = self.value_estimates[arm_index]
        self.value_estimates[arm_index] = estimate + (reward - estimate) / count

        # If all arms have been played at least once, we can start eliminating suboptimal arms
        active_arms = np.where(self.active_arms)[0]
        active_arm_counts = self.counts[active_arms]
        max_count = np.max(
            active_arm_counts
        )  # Should be one greater than the minimum count among active arms
        if len(active_arms[active_arm_counts < max_count]) == 0: # All arms have been played the same number of times, so we can start elimination
            # Deactivate arms whose UCB is less than max LCB of active arms
            active_arms = np.where(self.active_arms)[0]
            confidence_radius = np.sqrt(
                2 * np.log(self._run_horizon) / self.counts[active_arms]
            )
            ucb = self.value_estimates[active_arms] + confidence_radius
            lcb = self.value_estimates[active_arms] - confidence_radius
            max_lcb = np.max(lcb)
            self.active_arms[active_arms] &= (
                ucb >= max_lcb
            )  # We keep arms whose UCB is at least max LCB, and deactivate the rest
