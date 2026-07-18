from abc import abstractmethod
from dataclasses import field

import jax.numpy as jnp
import numpyro.distributions as dist
from gpjax.dataset import Dataset
from gpjax.gps import AbstractMeanFunction
from gpjax.typing import KeyArray
from jaxtyping import (
    Array,
    Float,
    Num,
)

from decijax.search_space import ContinuousSearchSpace


class AbstractContinuousTestFunction(AbstractMeanFunction):
    """
    Abstract base class for continuous test functions.

    Attributes:
        search_space (ContinuousSearchSpace): Search space for the function.
        maximizer (Float[Array, '1 D']): Maximizer of the function (to 5 decimal places)
        maximum (Float[Array, '1 1']): Maximum of the function (to 5 decimal places).
    """

    search_space: ContinuousSearchSpace
    maximizer: Float[Array, "1 D"]
    maximum: Float[Array, "1 1"]

    def generate_dataset(
        self, num_points: int, key: KeyArray, obs_stddev: float = 0.0
    ) -> Dataset:
        """
        Generate a toy dataset from the test function.

        Args:
            num_points (int): Number of points to sample.
            key (KeyArray): JAX PRNG key.
            obs_stddev (float): (Optional) standard deviation of Gaussian distributed
            noise added to observations.

        Returns:
            Dataset: Dataset of points sampled from the test function.
        """
        X = self.search_space.sample(num_points=num_points, key=key)
        gaussian_noise = dist.Normal(
            jnp.zeros(num_points), obs_stddev * jnp.ones(num_points)
        )
        y = self.evaluate(X) + jnp.transpose(
            gaussian_noise.sample(key, sample_shape=(1,))
        )
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

    def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N 1"]:
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        """
        Evaluate the test function at a set of points.

        Args:
            x (Float[Array, 'N D']): Points to evaluate the test function at.

        Returns:
            Float[Array, 'N 1']: Values of the test function at the points.
        """
        raise NotImplementedError


class NegativeForrester(AbstractContinuousTestFunction):
    """
    Negated Forrester function. The original Forrester function was introduced in
    'Engineering design via surrogate modelling: a practical guide' (Forrester et al.
    2008), rescaled to have zero mean and unit variance over $[0, 1]$. This class
    returns the negation of the original function, turning it into a maximisation
    problem.
    """

    search_space: ContinuousSearchSpace = field(
        default_factory=lambda: ContinuousSearchSpace(
            lower_bounds=jnp.array([0.0]),
            upper_bounds=jnp.array([1.0]),
        )
    )
    maximizer: Float[Array, "1 D"] = field(
        default_factory=lambda: jnp.array([[0.75725]])
    )
    maximum: Float[Array, "1 1"] = field(default_factory=lambda: jnp.array([[1.45280]]))

    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        mean = 0.45321
        std = jnp.sqrt(19.8577)
        return -(((6 * x - 2) ** 2) * jnp.sin(12 * x - 4) - mean) / std


class NegativeLogarithmicGoldsteinPrice(AbstractContinuousTestFunction):
    """
    Negated Logarithmic Goldstein-Price function. The original was introduced in 'A
    benchmark of kriging-based infill criteria for noisy optimization' (Picheny et al.
    2013), which has zero mean and unit variance over $[0, 1]^2$. This class returns
    the negation of the original function, turning it into a maximisation problem.
    """

    search_space: ContinuousSearchSpace = field(
        default_factory=lambda: ContinuousSearchSpace(
            lower_bounds=jnp.array([0.0, 0.0]),
            upper_bounds=jnp.array([1.0, 1.0]),
        )
    )
    maximizer: Float[Array, "1 D"] = field(
        default_factory=lambda: jnp.array([[0.5, 0.25]])
    )
    maximum: Float[Array, "1 1"] = field(default_factory=lambda: jnp.array([[3.12913]]))

    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        x1 = 4.0 * x[:, 0] - 2.0
        x2 = 4.0 * x[:, 1] - 2.0
        a = 1.0 + (x1 + x2 + 1.0) ** 2 * (
            19.0 - 14.0 * x1 + 3.0 * (x1**2) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * (x2**2)
        )
        b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (
            18.0
            - 32.0 * x1
            + 12.0 * (x1**2)
            + 48.0 * x2
            - 36.0 * x1 * x2
            + 27.0 * (x2**2)
        )
        return -((jnp.log((a * b)) - 8.693) / 2.427).reshape(-1, 1)


class NegativeQuadratic(AbstractContinuousTestFunction):
    """
    Negated toy quadratic function defined over $[0, 1]$. Has a maximum of 0.0 at
    $x = 0.5$.
    """

    search_space: ContinuousSearchSpace = field(
        default_factory=lambda: ContinuousSearchSpace(
            lower_bounds=jnp.array([0.0]),
            upper_bounds=jnp.array([1.0]),
        )
    )
    maximizer: Float[Array, "1 D"] = field(default_factory=lambda: jnp.array([[0.5]]))
    maximum: Float[Array, "1 1"] = field(default_factory=lambda: jnp.array([[0.0]]))

    def evaluate(self, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        return -((x - 0.5) ** 2)
