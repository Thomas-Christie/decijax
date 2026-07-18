from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from gpjax.dataset import Dataset
from gpjax.typing import (
    Array,
    Float,
    Int,
    KeyArray,
)

from decijax.search_space import ContinuousSearchSpace


@dataclass
class PoissonTestFunction:
    """
    Test function for GPs utilising the Poisson likelihood. Function taken from
    https://docs.jaxgaussianprocesses.com/_examples/poisson/#dataset.

    Attributes:
        search_space (ContinuousSearchSpace): Search space for the function.
    """

    search_space = ContinuousSearchSpace(
        lower_bounds=jnp.array([-2.0]),
        upper_bounds=jnp.array([2.0]),
    )

    def generate_dataset(self, num_points: int, key: KeyArray) -> Dataset:
        """
        Generate a toy dataset from the test function.

        Args:
            num_points (int): Number of points to sample.
            key (KeyArray): JAX PRNG key.

        Returns:
            Dataset: Dataset of points sampled from the test function.
        """
        X = self.search_space.sample(num_points=num_points, key=key)
        y = self.evaluate(X)
        return Dataset(X=X, y=y)

    def generate_test_points(
        self, num_points: int, key: KeyArray
    ) -> Float[Array, "N D"]:
        """
        Generate test points from the search space of the test function.

        Args:
            num_points (int): Number of points to sample.
            key (KeyArray): JAX PRNG key.

        Returns:
            Float[Array, 'N D']: Test points sampled from the search space.
        """
        return self.search_space.sample(num_points=num_points, key=key)

    @abstractmethod
    def evaluate(self, x: Float[Array, "N 1"]) -> Int[Array, "N 1"]:
        """
        Evaluate the test function at a set of points. Function taken from
        https://docs.jaxgaussianprocesses.com/_examples/poisson/#dataset.

        Args:
            x (Float[Array, 'N D']): Points to evaluate the test function at.

        Returns:
            Float[Array, 'N 1']: Values of the test function at the points.
        """
        key = jr.key(42)
        f = lambda x: 2.0 * jnp.sin(3 * x) + 0.5 * x
        return jr.poisson(key, jnp.exp(f(x)))
