from gpjax.gps import Prior
from gpjax.kernels.stationary.rbf import RBF
from gpjax.likelihoods import Gaussian
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)
from decijax.test_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.utils import (
    OBJECTIVE,
    build_function_evaluator,
    get_best_latent_observation_val,
)


def test_build_function_evaluator():
    def _square(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return x**2

    def _cube(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return x**3

    functions = {OBJECTIVE: _square, "CONSTRAINT": _cube}
    fn_evaluator = build_function_evaluator(functions)
    x = jnp.array([[2.0, 3.0]])
    datasets = fn_evaluator(x)
    assert datasets.keys() == functions.keys()
    assert jnp.equal(datasets[OBJECTIVE].X, x).all()
    assert jnp.equal(datasets[OBJECTIVE].y, _square(x)).all()
    assert jnp.equal(datasets["CONSTRAINT"].X, x).all()
    assert jnp.equal(datasets["CONSTRAINT"].y, _cube(x)).all()


@pytest.mark.parametrize(
    "test_target_function",
    [(NegativeForrester()), (NegativeLogarithmicGoldsteinPrice())],
)
@pytest.mark.parametrize("key", [jr.PRNGKey(42), jr.PRNGKey(10)])
def test_get_best_observation(
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    obs_stddev = 0.5
    dataset = test_target_function.generate_dataset(
        num_points=100, key=key, obs_stddev=obs_stddev
    )
    mean_fn = test_target_function
    kernel = RBF(lengthscale=1.0 * jnp.ones(dataset.X.shape[1]))
    prior = Prior(kernel=kernel, mean_function=mean_fn)
    likelihood = Gaussian(num_datapoints=dataset.n, obs_stddev=obs_stddev)
    posterior = prior * likelihood
    expected_best_obs = jnp.max(posterior(dataset.X, dataset).mean)
    actual_best_obs = get_best_latent_observation_val(
        posterior=posterior, dataset=dataset
    )
    assert jnp.equal(expected_best_obs, actual_best_obs)
