"""Abstract base classes and type aliases for acquisition functions."""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from typing import TypeAlias

from jaxtyping import (
    Array,
    Float,
)

from decijax.models import ProbabilisticModel
from decijax.typing import KeyArray

SinglePointAcquisitionFunction: TypeAlias = Callable[
    [Float[Array, "N D"]], Float[Array, "N 1"]
]
"""
Type alias for acquisition functions which don't support batching, and instead
characterise the utility of querying a single point, rather than a batch of points.
They take an array of points of shape $[N, D]$ and return the value of the acquisition
function at each point in an array of shape $[N, 1]$.
"""


AcquisitionFunction: TypeAlias = SinglePointAcquisitionFunction
"""
Type alias for all acquisition functions. Currently we only support
`SinglePointAcquisitionFunction`s, but in future may support batched acquisition
functions too. Note that `AcquisitionFunction`s are *maximised* in order to decide
which point, or batch of points, to query next.
"""


class AbstractSingleModelAcquisitionFunctionBuilder(ABC):
    """Abstract class for building acquisition functions which require a single model."""

    @abstractmethod
    def build_acquisition_function(
        self,
        model: ProbabilisticModel,
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        """Build an `AcquisitionFunction` from a model.

        Args:
            model: Model used to form the acquisition function, carrying the data it
                was conditioned on.
            key: JAX PRNG key used for random number generation.

        Returns:
            Acquisition function to be *maximised* in order to decide which point to
            query next.
        """
        raise NotImplementedError


AbstractAcquisitionFunctionBuilder = AbstractSingleModelAcquisitionFunctionBuilder
"""
Type alias for acquisition function builders. For now this only includes single model
acquisition function builders, but in the future we may support more acquisition
function builders.
"""
