from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from gpjax.typing import KeyArray
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.utility_functions.expected_improvement import (
    ExpectedImprovement,
)
from decijax.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)

from tests.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
def test_expected_improvement_utility_function_correct_values(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # Test validity of computed values with Monte-Carlo
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    ei_fn = ExpectedImprovement().build_utility_function(posteriors, datasets, key)
    test_x = test_target_function.generate_test_points(100, key)
    ei = ei_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    pred_dist = posterior.likelihood(latent_dist)
    pred_mean = pred_dist.mean
    pred_var = pred_dist.variance
    samples = dist.Normal(loc=pred_mean, scale=jnp.sqrt(pred_var)).sample(
        key, sample_shape=(10000,)
    )
    eta = get_best_latent_observation_val(posterior, dataset)
    mc_ei = jnp.expand_dims(jnp.mean(jnp.maximum(samples - eta, 0), 0), -1)
    assert jnp.all(ei >= 0)
    assert jnp.allclose(ei, mc_ei, rtol=0.03, atol=1e-6)
