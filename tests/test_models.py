import jax.numpy as jnp
import jax.random as jr
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from decijax.models import GPJaxConjugateGP
from decijax.test_functions import PoissonTestFunction
from gpjax.gps import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Poisson
from gpjax.mean_functions import Zero
from jaxtyping import TypeCheckError


@pytest.mark.filterwarnings("ignore:y is not of type float64")
def test_non_conjugate_posterior_raises_error():
    key = jr.key(42)
    test_function = PoissonTestFunction()
    dataset = test_function.generate_dataset(num_points=10, key=key)
    kernel = RBF(lengthscale=jnp.ones(dataset.X.shape[1]))
    prior = Prior(kernel=kernel, mean_function=Zero())
    posterior = prior * Poisson(num_datapoints=dataset.n)
    with pytest.raises(
        (BeartypeCallHintParamViolation, TypeCheckError, ValueError),
        match="ConjugatePosterior",
    ):
        GPJaxConjugateGP(posterior=posterior, dataset=dataset)
