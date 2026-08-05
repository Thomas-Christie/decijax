import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from decijax.acquisition_functions.expected_improvement import (
    ExpectedImprovement,
    LogExpectedImprovement,
    _log_ei_helper,
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


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_log_expected_improvement_acquisition_function_correct_values(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # LogEI must be the log of the (marginalised) EI: exp(LogEI) should recover both
    # the analytic EI and its Monte-Carlo estimate.
    data_key, ei_acq_key, log_ei_acq_key, test_key, mc_key = jr.split(key, 5)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    models = {OBJECTIVE: model}
    ei_fn = ExpectedImprovement().build_acquisition_function(models, ei_acq_key)
    log_ei_fn = LogExpectedImprovement().build_acquisition_function(
        models, log_ei_acq_key
    )
    test_x = test_target_function.generate_test_points(100, test_key)
    ei = ei_fn(test_x)
    log_ei = log_ei_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    latent_mean = latent_dist.mean
    latent_var = latent_dist.variance
    samples = dist.Normal(loc=latent_mean, scale=jnp.sqrt(latent_var)).sample(
        mc_key, sample_shape=(10000,)
    )
    eta = get_best_latent_observation_val(model)
    mc_ei = jnp.expand_dims(jnp.mean(jnp.maximum(samples - eta, 0), 0), -1)
    assert log_ei.shape == (100, 1)
    assert jnp.all(jnp.isfinite(log_ei))
    assert jnp.allclose(jnp.exp(log_ei), ei, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(jnp.exp(log_ei), mc_ei, rtol=0.03, atol=1e-6)


def test_log_ei_helper_is_stable_in_the_tails():
    # The naive log(phi(z) + z * Phi(z)) underflows to -inf, gradient included, well
    # within the range a maximiser explores. Straddles both branch points, -1 and -1e6.
    z = jnp.array(
        [
            -1e100,
            -1e10,
            -1e6 - 1.0,
            -1e6 + 1.0,
            -1e3,
            -300.0,
            -100.0,
            -37.6,  # see test_log_ei_helper_is_accurate_where_erfcx_is_broken
            -30.0,
            -10.0,
            -3.0,
            -1.0 - 1e-6,
            -1.0 + 1e-6,
            0.0,
            1.0,
            10.0,
        ]
    )
    log_ei = _log_ei_helper(z)
    grad = jax.grad(lambda z_: jnp.sum(_log_ei_helper(z_)))(z)

    assert jnp.all(jnp.isfinite(log_ei))
    assert jnp.all(jnp.isfinite(grad))
    # LogEI is strictly increasing in the scaled improvement.
    assert jnp.all(grad > 0)
    assert jnp.all(jnp.diff(log_ei) > 0)

    # Where the naive form still has headroom in float64, the two must agree.
    naive_z = z[z >= -30.0]
    naive = jnp.log(
        jax.scipy.stats.norm.pdf(naive_z) + naive_z * jax.scipy.stats.norm.cdf(naive_z)
    )
    assert jnp.allclose(_log_ei_helper(naive_z), naive, rtol=1e-6, atol=1e-6)

    # And the naive form is genuinely unusable where the helper is not.
    deep_tail = jnp.array([-100.0, -1e3])
    assert jnp.all(
        jnp.log(
            jax.scipy.stats.norm.pdf(deep_tail)
            + deep_tail * jax.scipy.stats.norm.cdf(deep_tail)
        )
        == -jnp.inf
    )


def test_log_ei_helper_matches_asymptotic_expansion_in_the_tails():
    # Finite and monotone is not enough; the deep tail must be *correct*, and there
    # the naive form has underflowed and is no use as a reference. For large |z|,
    # h(z) = phi(z) * (1/z^2 - 3/z^4 + 15/z^6 - ...).
    z = -jnp.concatenate(
        [
            jnp.linspace(10.0, 1000.0, 400),
            jnp.array([1e4, 1e6, 1e10, 1e50]),  # straddles the -1e6 branch point
        ]
    )
    y = 1.0 / z**2
    series = 1.0 - 3.0 * y + 15.0 * y**2 - 105.0 * y**3 + 945.0 * y**4
    expected = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(y * series)

    # Compared in absolute terms, against values ranging down to -1e100.
    assert jnp.allclose(_log_ei_helper(z), expected, rtol=0.0, atol=1e-4)


def test_log_ei_helper_is_accurate_where_erfcx_is_broken():
    # The band where jax.scipy.special.erfcx returns 0.0 maps to z in
    # [-37.68, -37.54]. Going through it naively gives a silently wrong (finite,
    # monotone) value, so pin it down against a reworking that reintroduces the bug.
    z = jnp.linspace(-37.7, -37.5, 501)
    y = 1.0 / z**2
    series = 1.0 - 3.0 * y + 15.0 * y**2 - 105.0 * y**3 + 945.0 * y**4
    expected = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(y * series)

    assert jnp.allclose(_log_ei_helper(z), expected, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
def test_log_expected_improvement_is_jit_and_grad_safe(
    test_target_function: AbstractContinuousTestFunction,
):
    # The maximiser jits and differentiates the closure, so it must be pure.
    key = jr.key(42)
    data_key, acq_key, test_key = jr.split(key, 3)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    log_ei_fn = LogExpectedImprovement().build_acquisition_function(
        {OBJECTIVE: model}, acq_key
    )
    test_x = test_target_function.generate_test_points(10, test_key)

    jitted = jax.jit(log_ei_fn)(test_x)
    grad = jax.grad(lambda x: jnp.sum(log_ei_fn(x)))(test_x)

    assert jnp.allclose(jitted, log_ei_fn(test_x))
    assert jnp.all(jnp.isfinite(grad))
