import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from decijax.acquisition_functions.probability_of_improvement import (
    LogProbabilityOfImprovement,
    ProbabilityOfImprovement,
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
def test_probability_of_improvement_gives_correct_value_against_erf(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    data_key, acq_key, test_key = jr.split(key, 3)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    models = {OBJECTIVE: model}

    pi_acquisition_builder = ProbabilityOfImprovement()
    pi_acquisition = pi_acquisition_builder.build_acquisition_function(models, acq_key)

    test_X = test_target_function.generate_test_points(num_points=100, key=test_key)
    acquisition_values = pi_acquisition(test_X)

    # Computing the expected acquisition values
    predictive_dist = posterior(test_X, train_data=dataset)
    predictive_mean = predictive_dist.mean
    predictive_std = predictive_dist.stddev()

    # Computing best_y as the max. of the posterior predictive mean
    # over the training set.
    predictive_dist_for_training_data = posterior(dataset.X, train_data=dataset)
    best_y = predictive_dist_for_training_data.mean.max()

    # 1 - Gaussian CDF computed "by hand"
    x_ = (best_y - predictive_mean) / predictive_std
    expected_acquisition_values = 1 - 0.5 * (
        1 + jax.scipy.special.erf(x_ / jnp.sqrt(2))
    ).reshape(-1, 1)

    assert acquisition_values.shape == (100, 1)
    assert jnp.isclose(acquisition_values, expected_acquisition_values).all()


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_probability_of_improvement_acquisition_function_correct_values(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # Test validity of computed values with Monte-Carlo.
    data_key, acq_key, test_key, mc_key = jr.split(key, 4)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    models = {OBJECTIVE: model}
    pi_fn = ProbabilityOfImprovement().build_acquisition_function(models, acq_key)
    test_x = test_target_function.generate_test_points(100, test_key)
    pi = pi_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    latent_mean = latent_dist.mean
    latent_var = latent_dist.variance
    samples = dist.Normal(loc=latent_mean, scale=jnp.sqrt(latent_var)).sample(
        mc_key, sample_shape=(10000,)
    )
    eta = get_best_latent_observation_val(model)
    mc_pi = jnp.expand_dims(jnp.mean(samples > eta, 0), -1)
    assert jnp.all(pi >= 0)
    assert jnp.all(pi <= 1)
    assert jnp.allclose(pi, mc_pi, rtol=0.03, atol=1e-3)


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_log_probability_of_improvement_acquisition_function_correct_values(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # LogPI must be the log of the (marginalised) PI: exp(LogPI) should recover
    # both the analytic PI and its Monte-Carlo estimate.
    data_key, pi_acq_key, log_pi_acq_key, test_key, mc_key = jr.split(key, 5)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    models = {OBJECTIVE: model}
    pi_fn = ProbabilityOfImprovement().build_acquisition_function(models, pi_acq_key)
    log_pi_fn = LogProbabilityOfImprovement().build_acquisition_function(
        models, log_pi_acq_key
    )
    test_x = test_target_function.generate_test_points(100, test_key)
    pi = pi_fn(test_x)
    log_pi = log_pi_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    latent_mean = latent_dist.mean
    latent_var = latent_dist.variance
    samples = dist.Normal(loc=latent_mean, scale=jnp.sqrt(latent_var)).sample(
        mc_key, sample_shape=(10000,)
    )
    eta = get_best_latent_observation_val(model)
    mc_pi = jnp.expand_dims(jnp.mean(samples > eta, 0), -1)
    assert jnp.allclose(jnp.exp(log_pi), pi, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(jnp.exp(log_pi), mc_pi, rtol=0.03, atol=1e-6)
