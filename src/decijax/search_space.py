"""Search spaces."""

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from gpjax.typing import (
    Array,
    KeyArray,
)
from jaxtyping import Float
from scipy.stats.qmc import Sobol


@dataclass
class AbstractSearchSpace(ABC):
    """Abstract base class for search spaces.

    Search spaces are used to define domains for sampling and optimisation
    functionality in decijax.
    """

    @abstractmethod
    def sample(self, num_points: int, key: KeyArray):
        """Sample points from the search space.

        Args:
            num_points (int): Number of points to be sampled from the search space.
            key (KeyArray): JAX PRNG key.

        Returns:
            A batch of `num_points` points sampled from the search space.
        """
        raise NotImplementedError


@dataclass
class ContinuousSearchSpace(AbstractSearchSpace):
    """Used to bound the domain of continuous real functions of dimension $D$."""

    lower_bounds: Float[Array, " D"]
    upper_bounds: Float[Array, " D"]

    def __post_init__(self):
        """Perform post-initialisation validity checks.

        Raises:
            ValueError: If `lower_bounds` and `upper_bounds` have different dtypes
                or shapes, are empty, or if any lower bound exceeds its
                corresponding upper bound.
        """
        if not self.lower_bounds.dtype == self.upper_bounds.dtype:
            raise ValueError("Lower and upper bounds must have the same dtype.")
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        if self.lower_bounds.shape[0] == 0:
            raise ValueError("Lower and upper bounds cannot be empty")
        if not (self.lower_bounds <= self.upper_bounds).all():
            raise ValueError("Lower bounds must be less than upper bounds.")

    @property
    def dimensionality(self) -> int:
        """Dimensionality of the search space."""
        return self.lower_bounds.shape[0]

    def sample(self, num_points: int, key: KeyArray) -> Float[Array, "{num_points} D"]:
        """Sample points from the search space using a Sobol sequence.

        Args:
            num_points (int): Number of points to be sampled from the search space.
            key (KeyArray): JAX PRNG key.

        Returns:
            `num_points` points sampled from the search space using a Sobol sequence.
        """
        if num_points <= 0:
            raise ValueError("Number of points must be greater than 0.")

        seed = int(jax.random.bits(key, dtype=jnp.uint32))
        sampler = Sobol(d=self.dimensionality, scramble=True, rng=seed)
        initial_sample = jnp.array(sampler.random(num_points))
        return (
            self.lower_bounds + (self.upper_bounds - self.lower_bounds) * initial_sample
        )
