from beartype.typing import Mapping
from jaxtyping import (
    Array,
    Key,
)

from decijax.acquisition_functions.base import (
    AbstractSinglePointAcquisitionFunctionBuilder,
    SinglePointAcquisitionFunction,
)
from decijax.models import (
    ProbabilisticModel,
    SupportsSamplePaths,
)
from decijax.utils import OBJECTIVE


class ThompsonSampling(AbstractSinglePointAcquisitionFunctionBuilder):
    """
    Form an acquisition function by drawing an approximate sample from the posterior,
    using decoupled sampling as introduced in [Wilson et. al.
    (2020)](https://arxiv.org/abs/2002.09309). The sample is returned directly as the
    acquisition function, which is then *maximised* to find the next query point.

    Note that this is a single batch acquisition function, as it doesn't support
    classical batching. However, Thompson sampling can be used in a batched setting by
    drawing a batch of different samples from the GP posterior. This can be done by calling `build_acquisition_function` with different keys, an example of which can
    be seen in the `ask` method of the `AcquisitionDrivenDecisionMaker` class. The
    samples can then be optimised sequentially.
    """

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: Key[Array, ""],
    ) -> SinglePointAcquisitionFunction:
        """
        Draw an approximate sample path from the posterior of the objective model and
        return it as an acquisition function to be *maximised*.

        Args:
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support differentiable sample paths, as we sample from the
                objective posterior to form the acquisition function.
            key (Key[Array, ""]): JAX PRNG key used for random number generation. This
                can be changed to draw different samples.

        Returns:
            SinglePointAcquisitionFunction: An approximate sample path from the
                objective model posterior to be *maximised* in order to decide which
                point to query next.
        """
        self.check_objective_present(models)

        objective_model = models[OBJECTIVE]
        if not isinstance(objective_model, SupportsSamplePaths):
            raise ValueError(
                "Objective model must support differentiable sample paths to draw an "
                "approximate Thompson sample."
            )

        return objective_model.draw_sample_paths(num_samples=1, key=key)
