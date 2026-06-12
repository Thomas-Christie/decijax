from jax import config

from decijax.utility_functions.expected_improvement import (
    ExpectedImprovement,
)

config.update("jax_enable_x64", True)

from beartype.typing import Type
from gpjax.typing import KeyArray
import jax.random as jr
import pytest

from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
)
from decijax.utility_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)
from decijax.utility_functions.thompson_sampling import ThompsonSampling
from decijax.utils import OBJECTIVE
from tests.utils import (
    generate_dummy_conjugate_posterior,
    generate_dummy_non_conjugate_posterior,
)


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
def test_utility_function_no_objective_posterior_raises_error(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
):
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {"CONSTRAINT": posterior}
    datasets = {OBJECTIVE: dataset}
    with pytest.raises(ValueError):
        utility_function = utility_function_builder(**utility_function_kwargs)
        utility_function.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
def test_utility_function_no_objective_dataset_raises_error(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
):
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {"CONSTRAINT": dataset}
    with pytest.raises(ValueError):
        utility_function = utility_function_builder(**utility_function_kwargs)
        utility_function.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
)
def test_non_conjugate_posterior_raises_error(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
):
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_non_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    with pytest.raises(ValueError):
        utility_function = utility_function_builder(**utility_function_kwargs)
        utility_function.build_utility_function(
            posteriors=posteriors, datasets=datasets, key=key
        )


@pytest.mark.parametrize(
    "utility_function_builder, utility_function_kwargs",
    [
        (ExpectedImprovement, {}),
        (ProbabilityOfImprovement, {}),
        (ThompsonSampling, {"num_features": 100}),
    ],
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
def test_utility_functions_have_correct_shapes(
    utility_function_builder: Type[AbstractSinglePointUtilityFunctionBuilder],
    utility_function_kwargs: dict,
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}
    utility_builder = utility_function_builder(**utility_function_kwargs)
    utility_function = utility_builder.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=key
    )
    test_key, _ = jr.split(key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    utility_function_values = utility_function(test_X)
    assert utility_function_values.shape == (num_test_points, 1)
