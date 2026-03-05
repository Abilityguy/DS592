"""Concrete n-armed Gaussian bandit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from bandit_sim.bandits.base import BanditEnvironment


@dataclass
class NArmedGaussianBandit(BanditEnvironment):
    """An N-armed Gaussian Bandit problem."""

    arm_means: tuple[float, ...]
    arm_stds: tuple[float, ...]
    seed: int | None = None
    _arm_means: npt.NDArray[np.float64] = field(init=False, repr=False)
    _arm_stds: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arm_means:
            raise ValueError("NArmedGaussianBandit requires at least one arm.")
        if len(self.arm_means) != len(self.arm_stds):
            raise ValueError("Arm means and standard deviations must have the same length.")

        self._arm_means = np.asarray(self.arm_means, dtype=np.float64)
        self._arm_stds = np.asarray(self.arm_stds, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_arms(self) -> int:
        return int(self._arm_means.shape[0])

    @property
    def expected_rewards(self) -> npt.NDArray[np.float64]:
        return self._arm_means.copy()

    def pull(self, arm_index: int) -> float:
        if arm_index < 0 or arm_index >= self.n_arms:
            raise IndexError("arm_index is out of range.")

        mean = self._arm_means[arm_index]
        std = self._arm_stds[arm_index]
        return float(self._rng.normal(loc=mean, scale=std))
