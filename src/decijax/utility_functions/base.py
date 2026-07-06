from abc import (
    ABC,
    abstractmethod,
)
from typing import TypeAlias

from beartype.typing import (
    Callable,
    Mapping,
)
from jaxtyping import (
    Array,
    Float,
    Key,
)

from decijax.models import ProbabilisticModel
from decijax.utils import OBJECTIVE

SinglePointUtilityFunction: TypeAlias = Callable[
    [Float[Array, "N D"]], Float[Array, "N 1"]
]
"""
Type alias for utility functions which don't support batching, and instead characterise
the utility of querying a single point, rather than a batch of points. They take an array of points of shape $[N, D]$
and return the value of the utility function at each point in an array of shape $[N, 1]$.
"""


UtilityFunction: TypeAlias = SinglePointUtilityFunction
"""
Type alias for all utility functions. Currently we only support
`SinglePointUtilityFunction`s, but in future may support batched utility functions too.
Note that `UtilityFunction`s are *maximised* in order to decide which point, or batch of points, to query next.
"""


class AbstractSinglePointUtilityFunctionBuilder(ABC):
    """
    Abstract class for building utility functions which don't support batches. As such,
    they characterise the utility of querying a single point next.
    """

    def check_objective_present(
        self,
        models: Mapping[str, ProbabilisticModel],
    ) -> None:
        """
        Check that the objective model is present in the models.

        Args:
            models: dictionary of models to be used to form the utility function.

        Raises:
            ValueError: If the objective model is not present in the models.
        """
        if OBJECTIVE not in models.keys():
            raise ValueError("Objective model not found in models")

    @abstractmethod
    def build_utility_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: Key[Array, ""],
    ) -> SinglePointUtilityFunction:
        """
        Build a `UtilityFunction` from a set of models.

        Args:
            models: dictionary of models to be used to form the utility function. Each
                model carries the data it was conditioned on.
            key: JAX PRNG key used for random number generation.

        Returns:
            SinglePointUtilityFunction: Utility function to be *maximised* in order to
                decide which point to query next.
        """
        raise NotImplementedError


AbstractUtilityFunctionBuilder = AbstractSinglePointUtilityFunctionBuilder
"""
Type alias for utility function builders. For now this only include single point utility
function builders, but in the future we may support batched utility function builders.
"""
