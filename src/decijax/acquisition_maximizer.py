"""Functionality for maximizing acquisition functions."""

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp
from jaxtyping import Array, Float
from scipy.optimize import minimize

from decijax.acquisition_functions import SinglePointAcquisitionFunction
from decijax.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)
from decijax.typing import KeyArray


def _get_top_k_query_points(
    query_points: Float[Array, "N D"],
    acquisition_function: SinglePointAcquisitionFunction,
    k: int,
) -> Float[Array, "K D"]:
    """Get the `k` points with the highest acquisition function values.

    Args:
        query_points: set of points at which to evaluate the acquisition function.
        acquisition_function: the acquisition function to be evaluated at
            `query_points`.
        k: number of points to return. Must be no greater than the number of
            `query_points`.

    Returns:
        Array of the `k` best points, ordered by decreasing acquisition function value.
    """
    acquisition_function_values = acquisition_function(query_points)
    _, top_k_indices = jax.lax.top_k(acquisition_function_values[:, 0], k)
    return query_points[top_k_indices]


@dataclass
class AbstractSinglePointAcquisitionMaximizer(ABC):
    """Abstract base class for single point acquisition function maximizers."""

    @abstractmethod
    def maximize(
        self,
        acquisition_function: SinglePointAcquisitionFunction,
        search_space: AbstractSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximize the given acquisition function over the search space provided.

        Args:
            acquisition_function: acquisition function to be maximized.
            search_space: search space over which to maximize the acquisition function.
            key: JAX PRNG key.

        Returns:
            Point at which the acquisition function is maximized.
        """
        raise NotImplementedError


@dataclass
class ContinuousSinglePointAcquisitionMaximizer(
    AbstractSinglePointAcquisitionMaximizer
):
    """Maximize acquisition functions over the continuous domain with L-BFGS-B.

    First we evaluate the acquisition function at `num_initial_samples` points sampled
    from the search space, and then we run L-BFGS-B from each of the best
    `num_optimization_runs` of these initial points, returning the best maximizer found
    across these runs.
    """

    num_initial_samples: int
    num_optimization_runs: int

    def __post_init__(self):
        """Validate `num_initial_samples` and `num_optimization_runs`.

        Raises:
            ValueError: If `num_initial_samples` or `num_optimization_runs` is less
                than 1, or if `num_optimization_runs` exceeds `num_initial_samples`.
        """
        if self.num_initial_samples < 1:
            raise ValueError(
                f"num_initial_samples must be greater than 0, got {self.num_initial_samples}."
            )
        elif self.num_optimization_runs < 1:
            raise ValueError(
                f"num_optimization_runs must be greater than 0, got {self.num_optimization_runs}."
            )
        elif self.num_optimization_runs > self.num_initial_samples:
            raise ValueError(
                "num_optimization_runs must be no greater than num_initial_samples, "
                f"got num_optimization_runs={self.num_optimization_runs} and "
                f"num_initial_samples={self.num_initial_samples}."
            )

    def maximize(
        self,
        acquisition_function: SinglePointAcquisitionFunction,
        search_space: ContinuousSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximize the acquisition function with multi-start L-BFGS-B.

        Samples `num_initial_samples` points from the search space, then runs L-BFGS-B
        from each of the best `num_optimization_runs` of them, and returns the best
        maximizer found across these runs.

        Args:
            acquisition_function: acquisition function to be maximized.
            search_space: continuous search space to maximize over.
            key: JAX PRNG key.

        Returns:
            Point at which the acquisition function is maximized.
        """
        initial_sample_points = search_space.sample(self.num_initial_samples, key=key)
        starting_points = _get_top_k_query_points(
            initial_sample_points, acquisition_function, self.num_optimization_runs
        )

        def _scalar_acquisition_function(
            x: Float[Array, "1 D"],
        ) -> Float[Array, ""]:
            """Returns the negative of the acquisition function as a scalar.

            This is because acquisition functions should be *maximized* but scipy
            *minimizes*.
            """
            return -acquisition_function(x)[0][0]

        val_and_grad_fn = jax.value_and_grad(_scalar_acquisition_function)

        def _objective_for_scipy(x_flat):
            x = jnp.array(x_flat).reshape(1, -1)
            val, grad = val_and_grad_fn(x)
            return float(val), onp.array(grad.flatten(), dtype=onp.float64)

        bounds = list(
            zip(
                onp.array(search_space.lower_bounds),
                onp.array(search_space.upper_bounds),
                strict=True,
            )
        )

        max_observed_acquisition_function_value = None
        maximizer = None
        for starting_point in starting_points:
            result = minimize(
                _objective_for_scipy,
                x0=onp.array(starting_point, dtype=onp.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            optimized_point = jnp.array(result.x).reshape(1, -1)
            optimized_acquisition_function_value = -result.fun
            if (max_observed_acquisition_function_value is None) or (
                optimized_acquisition_function_value
                > max_observed_acquisition_function_value
            ):
                max_observed_acquisition_function_value = (
                    optimized_acquisition_function_value
                )
                maximizer = optimized_point
        return maximizer


AbstractAcquisitionMaximizer = AbstractSinglePointAcquisitionMaximizer
"""
Type alias for an acquisition maximizer. Currently we only support single point
acquisition functions, but in future may support batched acquisition functions.
"""
