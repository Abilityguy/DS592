"""Linear contextual bandit with sub-Gaussian noise."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from bandit_sim.bandits.base import ContextualActionSet, ContextualBanditEnvironment, FiniteActionSet


ContextSampler = Callable[[np.random.Generator], ContextualActionSet | npt.NDArray[np.float64]]


@dataclass
class LinearSubGaussianBandit(ContextualBanditEnvironment):
    """A contextual linear bandit with configurable sub-Gaussian noise.

    At each round, the environment exposes an action set, which may either be a
    finite collection of action vectors or a continuous action set such as the
    unit ball. Rewards are generated according to

        r_t(a) = x_t(a)^T theta_star + noise_t

    where the noise is sampled from a concrete sub-Gaussian family.
    """

    context_dimension_: int
    theta_star: npt.NDArray[np.float64]
    context_sampler: ContextSampler
    noise_type: Literal["gaussian", "rademacher", "uniform"] = "gaussian"
    noise_scale: float = 1.0
    seed: int | None = None
    _theta_star: npt.NDArray[np.float64] = field(init=False, repr=False)
    _current_action_set: ContextualActionSet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.context_dimension_ <= 0:
            raise ValueError("context_dimension_ must be positive.")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative.")

        self._theta_star = np.asarray(self.theta_star, dtype=np.float64)
        if self._theta_star.shape != (self.context_dimension_,):
            raise ValueError(
                "theta_star must have shape "
                f"({self.context_dimension_},), got {self._theta_star.shape}."
            )

        self._rng = np.random.default_rng(self.seed)
        self._current_action_set = self._sample_and_validate_context()

    @property
    def context_dimension(self) -> int:
        return self.context_dimension_

    @property
    def action_set(self) -> ContextualActionSet:
        return self._current_action_set

    def sample_context(self) -> ContextualActionSet:
        self._current_action_set = self._sample_and_validate_context()
        return self.action_set

    def expected_reward(self, action: npt.NDArray[np.float64]) -> float:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.context_dimension_,):
            raise ValueError(
                f"action must have shape ({self.context_dimension_},), got {action_array.shape}."
            )
        if not self._current_action_set.contains(action_array):
            raise ValueError("action is not contained in the current action set.")
        return float(np.dot(action_array, self._theta_star))

    @property
    def best_expected_reward(self) -> float:
        return self._current_action_set.max_linear_value(self._theta_star)

    def pull(self, action: npt.NDArray[np.float64]) -> float:
        mean_reward = self.expected_reward(action)
        noise = self._sample_noise()
        return float(mean_reward + noise)

    def _sample_and_validate_context(self) -> ContextualActionSet:
        sampled_context = self.context_sampler(self._rng)
        if isinstance(sampled_context, ContextualActionSet):
            action_set = sampled_context
        else:
            action_features = np.asarray(sampled_context, dtype=np.float64)
            if action_features.ndim != 2:
                raise ValueError(
                    "context_sampler must return either a ContextualActionSet or "
                    "a 2D array with shape (n_actions, context_dimension)."
                )
            action_set = FiniteActionSet(action_features)

        if action_set.dimension != self.context_dimension_:
            raise ValueError(
                "Sampled action set has incompatible dimension "
                f"{action_set.dimension}; expected {self.context_dimension_}."
            )
        return action_set

    def _sample_noise(self) -> float:
        if self.noise_type == "gaussian":
            return float(self._rng.normal(loc=0.0, scale=self.noise_scale))
        if self.noise_type == "rademacher":
            return float(self.noise_scale * self._rng.choice(np.array([-1.0, 1.0], dtype=np.float64)))
        if self.noise_type == "uniform":
            return float(self._rng.uniform(low=-self.noise_scale, high=self.noise_scale))
        raise ValueError(
            "Invalid noise_type. Must be one of 'gaussian', 'rademacher', or 'uniform'."
        )
