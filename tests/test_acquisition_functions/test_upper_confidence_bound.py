import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from decijax.acquisition_functions.upper_confidence_bound import UpperConfidenceBound
from decijax.models import GPJaxConjugateGP
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.typing import KeyArray
from decijax.utils import OBJECTIVE

from tests.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize("beta", [0.0, 4.0])
@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_upper_confidence_bound_matches_closed_form(
    beta: float,
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # UCB is closed-form, so it can be checked exactly rather than statistically.
    # beta = 0 is included as the degenerate case: pure exploitation of the mean.
    data_key, acq_key, test_key = jr.split(key, 3)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    ucb_fn = UpperConfidenceBound(beta=beta).build_acquisition_function(
        {OBJECTIVE: model}, acq_key
    )
    test_x = test_target_function.generate_test_points(100, test_key)
    ucb = ucb_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    expected = latent_dist.mean + jnp.sqrt(beta) * jnp.sqrt(latent_dist.variance)

    assert ucb.shape == (100, 1)
    assert jnp.allclose(ucb, expected[:, None])


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_upper_confidence_bound_beta_weights_the_variance(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # beta weights the variance, so the bound sits sqrt(beta) standard deviations
    # above the mean and a fraction Phi(sqrt(beta)) of posterior samples falls below
    # it. This pins the parametrisation down: were beta the direct multiplier of
    # sigma, as it is in some other libraries, beta = 4 would cover 0.99997 here
    # rather than 0.97725.
    beta = 4.0
    data_key, acq_key, test_key, mc_key = jr.split(key, 4)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    ucb_fn = UpperConfidenceBound(beta=beta).build_acquisition_function(
        {OBJECTIVE: model}, acq_key
    )
    test_x = test_target_function.generate_test_points(100, test_key)
    ucb = ucb_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    samples = dist.Normal(
        loc=latent_dist.mean, scale=jnp.sqrt(latent_dist.variance)
    ).sample(mc_key, sample_shape=(10000,))
    coverage = jnp.mean(samples < ucb.squeeze(-1), axis=0)

    assert jnp.allclose(
        coverage, jax.scipy.stats.norm.cdf(jnp.sqrt(beta)), rtol=0.0, atol=0.01
    )


def test_upper_confidence_bound_negative_beta_raises_error():
    with pytest.raises(ValueError):
        UpperConfidenceBound(beta=-1.0)
