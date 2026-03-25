"""EXP3 algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.algorithms.base import BanditAlgorithm


@dataclass
class EXP3(BanditAlgorithm):
    """An implementation of the EXP3 algorithm."""

    seed: int | None = None
    weights: npt.NDArray[np.float64] = field(init=False)
    learning_rate: float = field(default=0.1)
    _run_horizon: int = field(default=0, init=False)
    probabilities: npt.NDArray[np.float64] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _initialize_state(self) -> None:
        assert self.n_arms is not None
        self.weights = np.ones(self.n_arms, dtype=np.float64)
        self.probabilities = None

    def _prepare_run(self, horizon: int) -> None:
        self._run_horizon = horizon

    def select_arm(self) -> int:
        self._check_initialized()
        assert self.n_arms is not None

        # P_t(i) = W_t(i) / W_t
        self.probabilities = self.weights / np.sum(self.weights, axis=None)

        return int(self._rng.choice(self.n_arms, p=self.probabilities))

    def update(self, arm_index: int, reward: float) -> None:
        self._check_initialized()
        assert self.probabilities is not None

        if arm_index < 0 or self.n_arms is None or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        # Compute an unbiased estimate of the reward
        # r~_t(i) = (1 - 1/P_t(i)) * (1 - r_t(i)) for the pulled arm
        unbiased_reward_estimate = (1 - 1 / self.probabilities[arm_index]) * (1 - reward)

        # Update weights
        self.weights[arm_index] *= np.exp(self.learning_rate * unbiased_reward_estimate)
