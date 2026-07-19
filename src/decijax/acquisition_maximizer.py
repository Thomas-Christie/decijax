"""Functionality for maximizing acquisition functions."""

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
    ScalarFloat,
)
from scipy.optimize import minimize

from decijax.acquisition_functions import SinglePointAcquisitionFunction
from decijax.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)


def _get_discrete_maximizer(
    query_points: Float[Array, "N D"],
    acquisition_function: SinglePointAcquisitionFunction,
) -> Float[Array, "1 D"]:
    """Get the point which maximises the acquisition function evaluated at a given set of points.

    Args:
        query_points: set of points at which to evaluate the acquisition function, as
            an array of shape `[n_points, n_dims]`.
        acquisition_function: the single point acquisition function to be evaluated at
            `query_points`.

    Returns:
        Array of shape `[1, n_dims]` representing the point which maximises the
        acquisition function.
    """
    acquisition_function_values = acquisition_function(query_points)
    max_acquisition_function_value_idx = jnp.argmax(
        acquisition_function_values, axis=0, keepdims=True
    )
    best_sample_point = jnp.take_along_axis(
        query_points, max_acquisition_function_value_idx, axis=0
    )
    return best_sample_point


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

    First we sample the acquisition function at `num_initial_samples` points from the
    search space, and then we run L-BFGS-B from the best of these initial points. We
    run this process `num_restarts` number of times, each time sampling a different
    random set of `num_initial_samples`initial points.
    """

    num_initial_samples: int
    num_restarts: int

    def __post_init__(self):
        """Validate that `num_initial_samples` and `num_restarts` are positive.

        Raises:
            ValueError: If `num_initial_samples` or `num_restarts` is less than 1.
        """
        if self.num_initial_samples < 1:
            raise ValueError(
                f"num_initial_samples must be greater than 0, got {self.num_initial_samples}."
            )
        elif self.num_restarts < 1:
            raise ValueError(
                f"num_restarts must be greater than 0, got {self.num_restarts}."
            )

    def maximize(
        self,
        acquisition_function: SinglePointAcquisitionFunction,
        search_space: ContinuousSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximize the acquisition function with multi-start L-BFGS-B.

        For each of `num_restarts` restarts, samples `num_initial_samples` points
        from the search space, seeds L-BFGS-B from the best of them, and returns
        the maximizer found across all restarts.

        Args:
            acquisition_function: acquisition function to be maximized.
            search_space: continuous search space to maximize over.
            key: JAX PRNG key.

        Returns:
            Point at which the acquisition function is maximized.
        """
        max_observed_acquisition_function_value = None
        maximizer = None

        for _ in range(self.num_restarts):
            key, _ = jr.split(key)
            initial_sample_points = search_space.sample(
                self.num_initial_samples, key=key
            )
            best_initial_sample_point = _get_discrete_maximizer(
                initial_sample_points, acquisition_function
            )

            def _scalar_acquisition_function(x: Float[Array, "1 D"]) -> ScalarFloat:
                """Returns the negative of the acquisition function as a scalar.

                This is because acquisition functions should be *maximized* but scipy
                *minimizes*.
                """
                return -acquisition_function(x)[0][0]

            val_and_grad_fn = jax.value_and_grad(_scalar_acquisition_function)

            def _objective_for_scipy(x_flat):
                x = jnp.array(x_flat).reshape(1, -1)
                val, grad = val_and_grad_fn(x)  # noqa: B023
                return float(val), onp.array(grad.flatten(), dtype=onp.float64)

            bounds = list(
                zip(
                    onp.array(search_space.lower_bounds),
                    onp.array(search_space.upper_bounds),
                    strict=True,
                )
            )
            result = minimize(
                _objective_for_scipy,
                x0=onp.array(best_initial_sample_point.flatten(), dtype=onp.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            optimized_point = jnp.array(result.x).reshape(1, -1)
            optimized_acquisition_function_value = acquisition_function(
                optimized_point
            )[0][0]
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
