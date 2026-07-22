from collections.abc import Callable

import jax.random as jr
import pytest
from decijax.acquisition_functions.thompson_sampling import ThompsonSampling
from decijax.models import GPJaxConjugateGP
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.typing import KeyArray
from decijax.utils import OBJECTIVE

from tests.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize("num_rff_features", [0, -1, -10])
def test_invalid_rff_num_on_model_raises_error(num_rff_features: int):
    # The number of random Fourier features is now owned by the model, so the
    # validation lives on the model rather than on ThompsonSampling.
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    with pytest.raises(ValueError):
        GPJaxConjugateGP(
            posterior=posterior, dataset=dataset, num_features=num_rff_features
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
def test_thompson_sampling_acquisition_function_same_key_same_function(
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset, num_features=100)
    models = {OBJECTIVE: model}
    ts_acquisition_function_one = ThompsonSampling().build_acquisition_function(
        models, key
    )
    ts_acquisition_function_two = ThompsonSampling().build_acquisition_function(
        models, key
    )
    test_key, _ = jr.split(key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    ts_acquisition_function_one_values = ts_acquisition_function_one(test_X)
    ts_acquisition_function_two_values = ts_acquisition_function_two(test_X)
    assert isinstance(ts_acquisition_function_one, Callable)
    assert isinstance(ts_acquisition_function_two, Callable)
    assert (
        ts_acquisition_function_one_values == ts_acquisition_function_two_values
    ).all()


@pytest.mark.parametrize(
    "test_target_function",
    [(NegativeForrester()), (NegativeLogarithmicGoldsteinPrice())],
)
@pytest.mark.parametrize("num_test_points", [50, 100])
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
def test_thompson_sampling_acquisition_function_different_key_different_function(
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset, num_features=100)
    models = {OBJECTIVE: model}
    sample_one_key = key
    sample_two_key, _ = jr.split(key)
    ts_acquisition_builder = ThompsonSampling()
    ts_acquisition_function_one = ts_acquisition_builder.build_acquisition_function(
        models, sample_one_key
    )
    ts_acquisition_function_two = ts_acquisition_builder.build_acquisition_function(
        models, sample_two_key
    )
    test_key, _ = jr.split(sample_two_key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    ts_acquisition_function_one_values = ts_acquisition_function_one(test_X)
    ts_acquisition_function_two_values = ts_acquisition_function_two(test_X)
    assert isinstance(ts_acquisition_function_one, Callable)
    assert isinstance(ts_acquisition_function_two, Callable)
    assert not (
        ts_acquisition_function_one_values == ts_acquisition_function_two_values
    ).all()
