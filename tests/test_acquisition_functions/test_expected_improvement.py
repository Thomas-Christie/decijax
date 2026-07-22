import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from decijax.acquisition_functions.expected_improvement import (
    ExpectedImprovement,
)
from decijax.models import GPJaxConjugateGP
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.typing import KeyArray
from decijax.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)

from tests.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_expected_improvement_acquisition_function_correct_values(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # Test validity of computed values with Monte-Carlo
    data_key, acq_key, test_key, mc_key = jr.split(key, 4)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    models = {OBJECTIVE: model}
    ei_fn = ExpectedImprovement().build_acquisition_function(models, acq_key)
    test_x = test_target_function.generate_test_points(100, test_key)
    ei = ei_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    latent_mean = latent_dist.mean
    latent_var = latent_dist.variance
    samples = dist.Normal(loc=latent_mean, scale=jnp.sqrt(latent_var)).sample(
        mc_key, sample_shape=(10000,)
    )
    eta = get_best_latent_observation_val(model)
    mc_ei = jnp.expand_dims(jnp.mean(jnp.maximum(samples - eta, 0), 0), -1)
    assert jnp.all(ei >= 0)
    assert jnp.allclose(ei, mc_ei, rtol=0.03, atol=1e-6)
