"""Upper Confidence Bound (UCB) algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms.base import BanditAlgorithm


@dataclass
class UpperConfidenceBound(BanditAlgorithm):
    """An implementation of the Upper Confidence Bound algorithm."""

    seed: int | None = None
    counts: npt.NDArray[np.int_] = field(init=False)
    value_estimates: npt.NDArray[np.float64] = field(init=False)
    delta: float = field(default=0.01, init=False)  # The confidence level for confidence intervals
    beta: float = field(default=1.0, init=False)  # The subgaussian parameter of the reward distributions
    _run_horizon: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _initialize_state(self) -> None:
        assert self.n_arms is not None
        self.counts = np.zeros(self.n_arms, dtype=np.int_)
        self.value_estimates = np.zeros(self.n_arms, dtype=np.float64)

    def _prepare_run(self, horizon: int) -> None:
        self._run_horizon = horizon

    def select_arm(self) -> int:
        self._check_initialized()
        assert self.n_arms is not None

        confidence_radius = self.beta * np.sqrt(
            2 * np.log(2 * self._run_horizon * self.n_arms / self.delta) / self.counts
        )
        ucb = self.value_estimates + confidence_radius

        return int(np.argmax(ucb))

    def update(self, arm_index: int, reward: float) -> None:
        self._check_initialized()

        if arm_index < 0 or self.n_arms is None or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        self.total_steps += 1
        self.counts[arm_index] += 1

        count = int(self.counts[arm_index])
        estimate = self.value_estimates[arm_index]
        self.value_estimates[arm_index] = estimate + (reward - estimate) / count
