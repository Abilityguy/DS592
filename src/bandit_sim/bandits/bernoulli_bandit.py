"""Bernoulli bandit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.bandits.base import BanditEnvironment


@dataclass
class BernoulliBandit(BanditEnvironment):
    """A Bernoulli Bandit problem."""

    arm_probs: tuple[float, ...]
    seed: int | None = None
    _arm_probs: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arm_probs:
            raise ValueError("BernoulliBandit requires at least one arm.")

        self._arm_probs = np.asarray(self.arm_probs, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_arms(self) -> int:
        return int(self._arm_probs.shape[0])

    @property
    def expected_rewards(self) -> npt.NDArray[np.float64]:
        return self._arm_probs.copy()

    def pull(self, arm_index: int) -> float:
        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        prob = self._arm_probs[arm_index]
        return float(self._bernoulli_sample(p=prob))

    def _bernoulli_sample(self, p: float) -> int:
        """Return 1 with probability p and 0 otherwise.

        Samples a uniform random variable and compares it against p,
        which is equivalent to drawing from Bernoulli(p).
        """
        return int(self._rng.uniform() < p)
