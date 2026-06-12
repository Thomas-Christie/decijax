from jax import config

config.update("jax_enable_x64", True)

from beartype.typing import Callable
from gpjax.typing import KeyArray
import jax.random as jr
import pytest

from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.utility_functions.thompson_sampling import ThompsonSampling
from decijax.utils import OBJECTIVE
from tests.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize("num_rff_features", [0, -1, -10])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_thompson_sampling_invalid_rff_num_raises_error(num_rff_features: int):
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    with pytest.raises(ValueError):
        ts_utility_builder = ThompsonSampling(num_features=num_rff_features)
        ts_utility_builder.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.parametrize(
    "test_target_function",
    [(NegativeForrester()), (NegativeLogarithmicGoldsteinPrice())],
)
@pytest.mark.parametrize("num_test_points", [50, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_thompson_sampling_utility_function_same_key_same_function(
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    ts_utility_builder_one = ThompsonSampling(num_features=100)
    ts_utility_builder_two = ThompsonSampling(num_features=100)
    ts_utility_function_one = ts_utility_builder_one.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    ts_utility_function_two = ts_utility_builder_two.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    test_key, _ = jr.split(key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    ts_utility_function_one_values = ts_utility_function_one(test_X)
    ts_utility_function_two_values = ts_utility_function_two(test_X)
    assert isinstance(ts_utility_function_one, Callable)
    assert isinstance(ts_utility_function_two, Callable)
    assert (ts_utility_function_one_values == ts_utility_function_two_values).all()


@pytest.mark.parametrize(
    "test_target_function",
    [(NegativeForrester()), (NegativeLogarithmicGoldsteinPrice())],
)
@pytest.mark.parametrize("num_test_points", [50, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_thompson_sampling_utility_function_different_key_different_function(
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    sample_one_key = key
    sample_two_key, _ = jr.split(key)
    ts_utility_builder = ThompsonSampling(num_features=100)
    ts_utility_function_one = ts_utility_builder.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=sample_one_key
    )
    ts_utility_function_two = ts_utility_builder.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=sample_two_key
    )
    test_key, _ = jr.split(sample_two_key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    ts_utility_function_one_values = ts_utility_function_one(test_X)
    ts_utility_function_two_values = ts_utility_function_two(test_X)
    assert isinstance(ts_utility_function_one, Callable)
    assert isinstance(ts_utility_function_two, Callable)
    assert not (ts_utility_function_one_values == ts_utility_function_two_values).all()
