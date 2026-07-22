from collections.abc import Mapping
from typing import Optional

import jax.numpy as jnp
from decijax.acquisition_functions import (
    AbstractSinglePointAcquisitionFunctionBuilder,
    SinglePointAcquisitionFunction,
)
from decijax.models import (
    GPJaxConjugateGP,
    ProbabilisticModel,
)
from decijax.test_functions import NegativeQuadratic
from decijax.typing import KeyArray
from gpjax.dataset import Dataset
from gpjax.gps import (
    ConjugatePosterior,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Zero,
)
from jaxtyping import (
    Array,
    Float,
)


class QuadraticSinglePointAcquisitionFunctionBuilder(
    AbstractSinglePointAcquisitionFunctionBuilder
):
    """Dummy acquisition function builder for testing purposes.

    Returns the value of the negated quadratic test function at the input points.
    The acquisition function is *maximised*, and the maximum is at x = 0.5.
    """

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        test_function = NegativeQuadratic()
        return test_function.evaluate


class CapabilitylessModel(ProbabilisticModel):
    """A model implementing no predictive capability.

    Used to test that acquisitions reject models which lack the capability they
    require.
    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @property
    def training_inputs(self) -> Float[Array, "N D"]:
        return self._dataset.X

    @property
    def observations(self) -> Float[Array, "N 1"]:
        return self._dataset.y


def generate_dummy_conjugate_posterior(
    dataset: Dataset,
    mean_function: Optional[AbstractMeanFunction] = None,
) -> ConjugatePosterior:
    kernel = RBF(lengthscale=jnp.ones(dataset.X.shape[1]))
    if mean_function is None:
        mean_function = Zero()
    prior = Prior(kernel=kernel, mean_function=mean_function)
    likelihood = Gaussian(num_datapoints=dataset.n, obs_stddev=1e-6)
    posterior = prior * likelihood
    return posterior


def generate_dummy_conjugate_model(
    dataset: Dataset,
    mean_function: Optional[AbstractMeanFunction] = None,
    num_features: int = 100,
) -> GPJaxConjugateGP:
    """Wrap a dummy conjugate GPJax posterior + dataset as a `GPJaxConjugateGP`."""
    posterior = generate_dummy_conjugate_posterior(dataset, mean_function)
    return GPJaxConjugateGP(
        posterior=posterior, dataset=dataset, num_features=num_features
    )
