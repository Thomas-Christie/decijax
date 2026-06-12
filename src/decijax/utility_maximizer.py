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

from decijax.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)
from decijax.utility_functions import SinglePointUtilityFunction


def _get_discrete_maximizer(
    query_points: Float[Array, "N D"], utility_function: SinglePointUtilityFunction
) -> Float[Array, "1 D"]:
    """Get the point which maximises the utility function evaluated at a given set of points.

    Args:
        query_points: set of points at which to evaluate the utility function, as an array
            of shape `[n_points, n_dims]`.
        utility_function: the single point utility function to be evaluated at `query_points`.

    Returns:
        Array of shape `[1, n_dims]` representing the point which maximises the utility function.
    """
    utility_function_values = utility_function(query_points)
    max_utility_function_value_idx = jnp.argmax(
        utility_function_values, axis=0, keepdims=True
    )
    best_sample_point = jnp.take_along_axis(
        query_points, max_utility_function_value_idx, axis=0
    )
    return best_sample_point


@dataclass
class AbstractSinglePointUtilityMaximizer(ABC):
    """Abstract base class for single point utility function maximizers."""

    @abstractmethod
    def maximize(
        self,
        utility_function: SinglePointUtilityFunction,
        search_space: AbstractSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        """Maximize the given utility function over the search space provided.

        Args:
            utility_function: utility function to be maximized.
            search_space: search space over which to maximize the utility function.
            key: JAX PRNG key.

        Returns:
            Float[Array, "1 D"]: Point at which the utility function is maximized.
        """
        raise NotImplementedError


@dataclass
class ContinuousSinglePointUtilityMaximizer(AbstractSinglePointUtilityMaximizer):
    """The `ContinuousUtilityMaximizer` class is used to maximize utility
    functions over the continuous domain with L-BFGS-B. First we sample the utility
    function at `num_initial_samples` points from the search space, and then we run
    L-BFGS-B from the best of these initial points. We run this process `num_restarts`
    number of times, each time sampling a different random set of
    `num_initial_samples`initial points.
    """

    num_initial_samples: int
    num_restarts: int

    def __post_init__(self):
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
        utility_function: SinglePointUtilityFunction,
        search_space: ContinuousSearchSpace,
        key: KeyArray,
    ) -> Float[Array, "1 D"]:
        max_observed_utility_function_value = None
        maximizer = None

        for _ in range(self.num_restarts):
            key, _ = jr.split(key)
            initial_sample_points = search_space.sample(
                self.num_initial_samples, key=key
            )
            best_initial_sample_point = _get_discrete_maximizer(
                initial_sample_points, utility_function
            )

            def _scalar_utility_function(x: Float[Array, "1 D"]) -> ScalarFloat:
                """
                Returns the negative of the utility function as a scalar, since
                utility functions should be *maximized* but scipy minimizes.
                """
                return -utility_function(x)[0][0]

            val_and_grad_fn = jax.value_and_grad(_scalar_utility_function)

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
            optimized_utility_function_value = utility_function(optimized_point)[0][0]
            if (max_observed_utility_function_value is None) or (
                optimized_utility_function_value > max_observed_utility_function_value
            ):
                max_observed_utility_function_value = optimized_utility_function_value
                maximizer = optimized_point
        return maximizer


AbstractUtilityMaximizer = AbstractSinglePointUtilityMaximizer
"""
Type alias for a utility maximizer. Currently we only support single point utility
functions, but in future may support batched utility functions.
"""
