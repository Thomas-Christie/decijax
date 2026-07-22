"""Abstract base classes and type aliases for acquisition functions."""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable, Mapping
from typing import TypeAlias

from jaxtyping import (
    Array,
    Float,
)

from decijax.models import ProbabilisticModel
from decijax.typing import KeyArray
from decijax.utils import OBJECTIVE

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


class AbstractSinglePointAcquisitionFunctionBuilder(ABC):
    """Abstract class for building acquisition functions which don't support batches.

    These acquisition functions characterise the utility of querying a single point next.
    """

    def check_objective_present(
        self,
        models: Mapping[str, ProbabilisticModel],
    ) -> None:
        """Check that the objective model is present in the models.

        Args:
            models: dictionary of models to be used to form the acquisition function.

        Raises:
            ValueError: If the objective model is not present in the models.
        """
        if OBJECTIVE not in models.keys():
            raise ValueError("Objective model not found in models")

    @abstractmethod
    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        """Build an `AcquisitionFunction` from a set of models.

        Args:
            models: dictionary of models to be used to form the acquisition function.
                Each model carries the data it was conditioned on.
            key: JAX PRNG key used for random number generation.

        Returns:
            SinglePointAcquisitionFunction: Acquisition function to be *maximised* in
                order to decide which point to query next.
        """
        raise NotImplementedError


AbstractAcquisitionFunctionBuilder = AbstractSinglePointAcquisitionFunctionBuilder
"""
Type alias for acquisition function builders. For now this only include single point
acquisition function builders, but in the future we may support batched acquisition
function builders.
"""
