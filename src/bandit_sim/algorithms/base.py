"""Abstract interface for bandit algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.bandits.base import BanditEnvironment


@dataclass
class SimulationResult:
    """Stores the outputs of a simulation run."""

    rewards: npt.NDArray[np.float64]
    actions: npt.NDArray[np.int_]

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def steps(self) -> int:
        return int(self.rewards.shape[0])


@dataclass
class SimulationBatchResult:
    """Stores the outputs of multiple simulation runs."""

    rewards: npt.NDArray[np.float64]
    actions: npt.NDArray[np.int_]

    @property
    def total_rewards(self) -> npt.NDArray[np.float64]:
        return self.rewards.sum(axis=1)

    @property
    def average_total_reward(self) -> float:
        if self.rewards.size == 0:
            return 0.0
        return float(self.total_rewards.mean())

    @property
    def n_simulations(self) -> int:
        return int(self.rewards.shape[0])


@dataclass
class BanditAlgorithm(ABC):
    """Base class for any algorithm that interacts with a bandit."""

    n_arms: int | None = field(default=None, init=False)
    total_steps: int = field(default=0, init=False)

    def initialize(self, n_arms: int) -> None:
        """Prepare the algorithm to interact with a bandit with `n_arms`."""
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")

        self.n_arms = n_arms
        self.total_steps = 0
        self._initialize_state()

    def run(self, bandit: BanditEnvironment, horizon: int) -> SimulationResult:
        """Run this algorithm against a bandit for a fixed number of steps."""
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        self.initialize(bandit.n_arms)
        self._prepare_run(horizon)

        rewards = np.empty(horizon, dtype=np.float64)
        actions = np.empty(horizon, dtype=np.int_)

        for step in range(horizon):
            arm_index = self.select_arm()
            reward = bandit.pull(arm_index)
            self.update(arm_index, reward)

            actions[step] = arm_index
            rewards[step] = reward

        return SimulationResult(rewards=rewards, actions=actions)

    def run_n_simulations(
        self,
        bandit: BanditEnvironment,
        horizon: int,
        n_simulations: int,
    ) -> SimulationBatchResult:
        """Run this algorithm against the same bandit multiple times."""
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive.")

        rewards = np.empty((n_simulations, horizon), dtype=np.float64)
        actions = np.empty((n_simulations, horizon), dtype=np.int_)

        for simulation_index in range(n_simulations):
            result = self.run(bandit=bandit, horizon=horizon)
            rewards[simulation_index] = result.rewards
            actions[simulation_index] = result.actions

        return SimulationBatchResult(rewards=rewards, actions=actions)

    def _prepare_run(self, horizon: int) -> None:
        """Allow subclasses to derive per-run state from the simulation horizon."""

    @abstractmethod
    def _initialize_state(self) -> None:
        """Reset any algorithm-specific state."""

    @abstractmethod
    def select_arm(self) -> int:
        """Choose which arm to pull next."""

    @abstractmethod
    def update(self, arm_index: int, reward: float) -> None:
        """Update internal state after observing a reward."""

    def _check_initialized(self) -> None:
        if self.n_arms is None:
            raise RuntimeError("Algorithm has not been initialized with a bandit.")
