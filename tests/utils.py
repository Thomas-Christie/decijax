from beartype.typing import (
    Mapping,
    Optional,
)
from gpjax.dataset import Dataset
from gpjax.gps import (
    ConjugatePosterior,
    NonConjugatePosterior,
    Prior,
)
from gpjax.kernels import RBF
from gpjax.likelihoods import (
    Gaussian,
    Poisson,
)
from gpjax.mean_functions import (
    AbstractMeanFunction,
    Zero,
)
from gpjax.typing import KeyArray
import jax.numpy as jnp

from decijax.test_functions import NegativeQuadratic
from decijax.utility_functions import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)


class QuadraticSinglePointUtilityFunctionBuilder(
    AbstractSinglePointUtilityFunctionBuilder
):
    """
    Dummy utility function builder for testing purposes, which returns the value of the
    negated quadratic test function at the input points. The utility function is
    *maximised*, and the maximum is at x = 0.5.
    """

    def build_utility_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        test_function = NegativeQuadratic()
        return test_function.evaluate


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


def generate_dummy_non_conjugate_posterior(
    dataset: Dataset,
    mean_function: Optional[AbstractMeanFunction] = None,
) -> NonConjugatePosterior:
    kernel = RBF(lengthscale=jnp.ones(dataset.X.shape[1]))
    if mean_function is None:
        mean_function = Zero()
    prior = Prior(kernel=kernel, mean_function=mean_function)
    likelihood = Poisson(num_datapoints=dataset.n)
    posterior = prior * likelihood
    return posterior
