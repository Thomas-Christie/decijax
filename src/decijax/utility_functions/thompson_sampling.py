from beartype.typing import Mapping
from jaxtyping import (
    Array,
    Key,
)

from decijax.models import (
    ProbabilisticModel,
    SupportsSamplePaths,
)
from decijax.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from decijax.utils import OBJECTIVE


class ThompsonSampling(AbstractSinglePointUtilityFunctionBuilder):
    """
    Form a utility function by drawing an approximate sample from the posterior,
    using decoupled sampling as introduced in [Wilson et. al.
    (2020)](https://arxiv.org/abs/2002.09309). The sample is returned directly as the
    utility function, which is then *maximised* to find the next query point.

    Note that this is a single batch utility function, as it doesn't support classical
    batching. However, Thompson sampling can be used in a batched setting by drawing a
    batch of different samples from the GP posterior. This can be done by calling
    `build_utility_function` with different keys, an example of which can be seen in the
    `ask` method of the `UtilityDrivenDecisionMaker` class. The samples can then be
    optimised sequentially.
    """

    def build_utility_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: Key[Array, ""],
    ) -> SinglePointUtilityFunction:
        """
        Draw an approximate sample path from the posterior of the objective model and
        return it as a utility function to be *maximised*.

        Args:
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
            the utility function. One model must correspond to the `OBJECTIVE` key and
            support differentiable sample paths, as we sample from the objective
            posterior to form the utility function.
            key (Key[Array, ""]): JAX PRNG key used for random number generation. This
            can be changed to draw different samples.

        Returns:
            SinglePointUtilityFunction: An approximate sample path from the objective
                model posterior to be *maximised* in order to decide which point to
                query next.
        """
        self.check_objective_present(models)

        objective_model = models[OBJECTIVE]
        if not isinstance(objective_model, SupportsSamplePaths):
            raise ValueError(
                "Objective model must support differentiable sample paths to draw an "
                "approximate Thompson sample."
            )

        return objective_model.draw_sample_paths(num_samples=1, key=key)
