"""Abstract interface for bandit environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class BanditEnvironment(ABC):
    """Base class for a bandit problem."""

    @property
    @abstractmethod
    def n_arms(self) -> int:
        """Return the number of available arms."""

    @abstractmethod
    def pull(self, arm_index: int) -> float:
        """Pull an arm and return the realized reward."""

    @property
    @abstractmethod
    def expected_rewards(self) -> npt.NDArray[np.float64]:
        """Return the expected reward of each arm."""

    @property
    def optimal_arm(self) -> int:
        """Return the index of the arm with the largest expected reward."""
        return int(np.argmax(self.expected_rewards))


@dataclass
class ContextualActionSet(ABC):
    """Base class for action sets used in contextual and linear bandits."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the ambient dimension of each action vector."""

    @abstractmethod
    def contains(self, action: npt.NDArray[np.float64]) -> bool:
        """Return whether the action belongs to this action set."""

    @abstractmethod
    def argmax_linear(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return an action maximizing the linear score a^T theta."""

    def max_linear_value(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the maximum linear value achievable in this action set."""
        best_action = self.argmax_linear(theta)
        return float(np.dot(best_action, theta))


@dataclass
class FiniteActionSet(ContextualActionSet):
    """A finite set of action vectors."""

    actions: npt.NDArray[np.float64]
    _actions: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._actions = np.asarray(self.actions, dtype=np.float64)
        if self._actions.ndim != 2:
            raise ValueError("FiniteActionSet expects a 2D array of shape (n_actions, dimension).")
        if self._actions.shape[0] == 0:
            raise ValueError("FiniteActionSet requires at least one action.")

    @property
    def dimension(self) -> int:
        return int(self._actions.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self._actions.shape[0])

    @property
    def action_features(self) -> npt.NDArray[np.float64]:
        """Return the available action vectors with shape (n_actions, dimension)."""
        return self._actions.copy()

    def contains(self, action: npt.NDArray[np.float64]) -> bool:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.dimension,):
            return False

        coordinate_matches = np.isclose(self._actions, action_array)
        row_matches = np.all(coordinate_matches, axis=1)
        has_matching_row = np.any(row_matches)
        return bool(has_matching_row)

    def argmax_linear(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        theta_array = np.asarray(theta, dtype=np.float64)
        if theta_array.shape != (self.dimension,):
            raise ValueError(
                f"theta must have shape ({self.dimension},), got {theta_array.shape}."
            )
        scores = self._actions @ theta_array
        return self._actions[int(np.argmax(scores))].copy()


@dataclass
class UnitBallActionSet(ContextualActionSet):
    """The Euclidean ball {a in R^d : ||a||_2 <= radius}."""

    dimension_: int
    radius: float = 1.0

    def __post_init__(self) -> None:
        if self.dimension_ <= 0:
            raise ValueError("dimension_ must be positive.")
        if self.radius <= 0:
            raise ValueError("radius must be positive.")

    @property
    def dimension(self) -> int:
        return self.dimension_

    def contains(self, action: npt.NDArray[np.float64]) -> bool:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.dimension_,):
            return False
        return bool(np.linalg.norm(action_array, ord=2) <= self.radius + 1e-12)

    def argmax_linear(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        theta_array = np.asarray(theta, dtype=np.float64)
        if theta_array.shape != (self.dimension_,):
            raise ValueError(
                f"theta must have shape ({self.dimension_},), got {theta_array.shape}."
            )
        norm = float(np.linalg.norm(theta_array, ord=2))
        if norm == 0.0: # Every action is optimal, so return the zero vector for simplicity.
            return np.zeros(self.dimension_, dtype=np.float64)
        
        # The optimal action is the point on the boundary of the ball in the direction of theta.
        return self.radius * theta_array / norm


@dataclass
class ContextualBanditEnvironment(ABC):
    """Base class for a contextual bandit problem.

    A contextual bandit exposes an action set at the current decision round.
    Algorithms choose an action vector from that set, then receive a reward
    sampled from the environment.
    """

    @property
    @abstractmethod
    def context_dimension(self) -> int:
        """Return the ambient dimension of each action vector."""

    @property
    @abstractmethod
    def action_set(self) -> ContextualActionSet:
        """Return the current round's feasible action set."""

    @abstractmethod
    def sample_context(self) -> ContextualActionSet:
        """Advance to the next round's context and return its action set."""

    @abstractmethod
    def pull(self, action: npt.NDArray[np.float64]) -> float:
        """Play an action vector and return the realized reward."""

    @abstractmethod
    def expected_reward(self, action: npt.NDArray[np.float64]) -> float:
        """Return the expected reward of the given action under the current context."""

    @property
    @abstractmethod
    def best_expected_reward(self) -> float:
        """Return the largest expected reward under the current context."""
