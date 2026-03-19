"""Abstract interface for bandit environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
