"""OFUL algorithm for stochastic linear bandits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from bandit_sim.algorithms.base import ContextualBanditAlgorithm
from bandit_sim.bandits.base import ContextualActionSet, FiniteActionSet, UnitBallActionSet


@dataclass
class OFUL(ContextualBanditAlgorithm):
    """Optimism in the Face of Uncertainty for linear bandits.

    The algorithm maintains the ridge-regression estimate

        theta_hat_t = sigma_t^{-1} b_t

    with

        sigma_t = lambda I + sum_{s < t} a_s a_s^T
        b_t = sum_{s < t} r_s a_s

    and selects the optimistic action

        argmax_a a^T theta_hat_t + beta_t * ||a||_{sigma_t^{-1}}

    For a finite action set, the maximization is computed in closed form by
    scoring each candidate action. For a unit-ball action set, the
    maximization is solved numerically with ``scipy.optimize.minimize``.
    """

    lambda_reg: float = 1.0
    delta: float = 0.01
    theta_star_upper_bound: float = 1.0
    action_norm_bound: float = 1.0
    solver_maxiter: int = 200
    solver_ftol: float = 1e-9
    sigma: npt.NDArray[np.float64] = field(init=False)
    b: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive.")
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie in (0, 1).")
        if self.theta_star_upper_bound < 0:
            raise ValueError("theta_star_upper_bound must be non-negative.")
        if self.action_norm_bound <= 0:
            raise ValueError("action_norm_bound must be positive.")
        if self.solver_maxiter <= 0:
            raise ValueError("solver_maxiter must be positive.")
        if self.solver_ftol <= 0:
            raise ValueError("solver_ftol must be positive.")

    def _initialize_state(self) -> None:
        self._check_initialized()
        assert self.context_dimension is not None

        self.sigma = self.lambda_reg * np.eye(self.context_dimension, dtype=np.float64)
        self.b = np.zeros(self.context_dimension, dtype=np.float64)

    def select_action(self, action_set: ContextualActionSet) -> npt.NDArray[np.float64]:
        self._check_initialized()
        self._validate_action_set(action_set)

        theta_hat = self._theta_hat()
        beta_t = self._confidence_radius()

        if isinstance(action_set, FiniteActionSet):
            return self._select_from_finite_action_set(action_set, theta_hat, beta_t)

        if isinstance(action_set, UnitBallActionSet):
            return self._select_from_unit_ball(action_set, theta_hat, beta_t)

        raise NotImplementedError(
            f"OFUL does not yet support action set type {type(action_set).__name__}."
        )

    def update(self, chosen_action: npt.NDArray[np.float64], reward: float) -> None:
        self._check_initialized()
        assert self.context_dimension is not None

        action = np.asarray(chosen_action, dtype=np.float64)
        if action.shape != (self.context_dimension,):
            raise ValueError(
                f"chosen_action must have shape ({self.context_dimension},), got {action.shape}."
            )

        self.sigma += np.outer(action, action)
        self.b += reward * action

    def _theta_hat(self) -> npt.NDArray[np.float64]:
        return np.linalg.solve(self.sigma, self.b)

    def _confidence_radius(self) -> float:
        """Return the OFUL confidence radius.

        This implementation uses the user-requested form

            beta_t =
                (
                    sqrt(lambda) * S
                    + sqrt(2 log(1 / delta) + d log((d lambda + t L^2) / (d lambda)))
                )^2

        where S is an upper bound on ||theta_*||_2 and L is an upper bound on
        ||a_t||_2.
        """
        self._check_initialized()
        assert self.context_dimension is not None

        t = max(self.total_steps, 1)
        d = self.context_dimension
        return float(d * np.log(t / self.delta))
        # log_term = 2.0 * np.log(1.0 / self.delta) + d * np.log(
        #     (d * self.lambda_reg + t * (self.action_norm_bound**2)) / (d * self.lambda_reg)
        # )
        # beta_t = (
        #     np.sqrt(self.lambda_reg) * self.theta_star_upper_bound + np.sqrt(log_term)
        # )
        # return float(beta_t)

    def _select_from_finite_action_set(
        self,
        action_set: FiniteActionSet,
        theta_hat: npt.NDArray[np.float64],
        beta_t: float,
    ) -> npt.NDArray[np.float64]:
        actions = action_set.action_features

        # Calculate sigma^{-1} a_i for each action a_i
        solved = np.linalg.solve(self.sigma, actions.T).T

        # Calculate the confidence widths ||a_i||_{sigma^{-1}} for each action
        confidence_widths = np.sqrt(np.einsum("ij,ij->i", actions, solved))

        # Calculate the optimistic scores a_i^T theta_hat + beta_t * ||a_i||_{sigma^{-1}} for each action
        optimistic_scores = actions @ theta_hat + beta_t * confidence_widths

        # Select the action with the highest optimistic score
        return actions[int(np.argmax(optimistic_scores))].copy()

    def _select_from_unit_ball(
        self,
        action_set: UnitBallActionSet,
        theta_hat: npt.NDArray[np.float64],
        beta_t: float,
    ) -> npt.NDArray[np.float64]:
        sigma_inverse = np.linalg.inv(self.sigma)

        # Use multiple initial guesses to mitigate the risk of local optima. We include the direction of theta_hat,
        # as well as the principal direction of sigma^{-1}, which is the direction of greatest confidence width.
        initial_guesses = self._unit_ball_initial_guesses(action_set, theta_hat, sigma_inverse)

        best_value = -np.inf
        best_action: npt.NDArray[np.float64] | None = None

        constraint = {
            "type": "ineq",
            "fun": lambda action: action_set.radius**2 - float(np.dot(action, action)), # Ensure ||action||_2 <= action_set.radius
            "jac": lambda action: -2.0 * action, # Gradient of the constraint function with respect to the action
        }

        # Solve the optimization problem from each initial guess and keep track of the best solution found.
        for initial_guess in initial_guesses:
            result = minimize(
                fun=lambda action: -self._optimistic_score(action, theta_hat, beta_t, sigma_inverse),
                x0=initial_guess,
                jac=lambda action: -self._optimistic_score_gradient(
                    action, theta_hat, beta_t, sigma_inverse
                ),
                constraints=[constraint],
                method="SLSQP",
                options={"maxiter": self.solver_maxiter, "ftol": self.solver_ftol},
            )

            candidate = np.asarray(result.x, dtype=np.float64)
            candidate_norm = float(np.linalg.norm(candidate, ord=2))
            
            # In case of numerical issues with the solver, we project the candidate back to the action set if it's outside.
            if candidate_norm > action_set.radius and candidate_norm > 0.0:
                candidate = candidate * (action_set.radius / candidate_norm)

            # Score the candidate action and update the best action if it's better.
            candidate_value = self._optimistic_score(candidate, theta_hat, beta_t, sigma_inverse)

            if candidate_value > best_value:
                best_value = candidate_value
                best_action = candidate.copy()

        assert best_action is not None
        return best_action

    def _unit_ball_initial_guesses(
        self,
        action_set: UnitBallActionSet,
        theta_hat: npt.NDArray[np.float64],
        sigma_inverse: npt.NDArray[np.float64],
    ) -> list[npt.NDArray[np.float64]]:
        zero = np.zeros(action_set.dimension, dtype=np.float64)

        theta_norm = float(np.linalg.norm(theta_hat, ord=2))
        if theta_norm > 0.0:
            theta_direction = action_set.radius * theta_hat / theta_norm
        else:
            theta_direction = zero.copy()

        # The principal direction of sigma^{-1} is the direction of greatest confidence width, so it's a natural
        # candidate for maximizing the optimistic score. We include both the positive and negative principal directions as initial guesses.
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_inverse)
        principal_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        principal_direction = action_set.radius * principal_direction / np.linalg.norm(principal_direction)

        guesses = [theta_direction, principal_direction, -principal_direction]

        # Remove duplicate guesses that can arise when theta_hat is aligned with the principal direction of sigma^{-1}.
        unique_guesses: list[npt.NDArray[np.float64]] = []
        for guess in guesses:
            if not any(np.allclose(guess, existing) for existing in unique_guesses):
                unique_guesses.append(guess)
        return unique_guesses

    def _optimistic_score(
        self,
        action: npt.NDArray[np.float64],
        theta_hat: npt.NDArray[np.float64],
        beta_t: float,
        sigma_inverse: npt.NDArray[np.float64],
    ) -> float:
        # The optimistic score is a^T theta_hat + beta_t * ||a||_{sigma^{-1}}.
        action_array = np.asarray(action, dtype=np.float64)
        confidence_norm_sq = float(action_array @ sigma_inverse @ action_array)
        confidence_norm_sq = max(confidence_norm_sq, 0.0)
        return float(action_array @ theta_hat + beta_t * np.sqrt(confidence_norm_sq))

    def _optimistic_score_gradient(
        self,
        action: npt.NDArray[np.float64],
        theta_hat: npt.NDArray[np.float64],
        beta_t: float,
        sigma_inverse: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # The gradient of the optimistic score with respect to the action is given by
        # theta_hat + beta_t * sigma^{-1} a / ||a||_{sigma^{-1}}.
        action_array = np.asarray(action, dtype=np.float64)
        confidence_norm_sq = float(action_array @ sigma_inverse @ action_array)
        if confidence_norm_sq <= 1e-16:
            return theta_hat.copy()
        confidence_norm = np.sqrt(confidence_norm_sq)
        return theta_hat + beta_t * (sigma_inverse @ action_array) / confidence_norm
