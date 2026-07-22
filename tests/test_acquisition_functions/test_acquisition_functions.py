import jax.random as jr
import pytest
from decijax.acquisition_functions.base import (
    AbstractSinglePointAcquisitionFunctionBuilder,
)
from decijax.acquisition_functions.expected_improvement import (
    ExpectedImprovement,
)
from decijax.acquisition_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)
from decijax.acquisition_functions.thompson_sampling import ThompsonSampling
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.typing import KeyArray
from decijax.utils import OBJECTIVE

from tests.utils import (
    CapabilitylessModel,
    generate_dummy_conjugate_model,
)


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "acquisition_function_builder",
    [ExpectedImprovement, ProbabilityOfImprovement, ThompsonSampling],
)
def test_acquisition_function_no_objective_model_raises_error(
    acquisition_function_builder: type[AbstractSinglePointAcquisitionFunctionBuilder],
):
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    model = generate_dummy_conjugate_model(dataset)
    models = {"CONSTRAINT": model}  # No OBJECTIVE-tagged model
    with pytest.raises(ValueError):
        acquisition_function = acquisition_function_builder()
        acquisition_function.build_acquisition_function(models, key)


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "acquisition_function_builder",
    [ExpectedImprovement, ProbabilityOfImprovement, ThompsonSampling],
)
def test_model_without_required_capability_raises_error(
    acquisition_function_builder: type[AbstractSinglePointAcquisitionFunctionBuilder],
):
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    model = CapabilitylessModel(dataset)  # Supports neither prediction nor sampling
    models = {OBJECTIVE: model}
    with pytest.raises(ValueError):
        acquisition_function = acquisition_function_builder()
        acquisition_function.build_acquisition_function(models, key)


@pytest.mark.parametrize(
    "acquisition_function_builder",
    [ExpectedImprovement, ProbabilityOfImprovement, ThompsonSampling],
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
def test_acquisition_functions_have_correct_shapes(
    acquisition_function_builder: type[AbstractSinglePointAcquisitionFunctionBuilder],
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    dataset = test_target_function.generate_dataset(num_points=10, key=key)
    model = generate_dummy_conjugate_model(dataset)
    models = {OBJECTIVE: model}
    acquisition_builder = acquisition_function_builder()
    acquisition_function = acquisition_builder.build_acquisition_function(models, key)
    test_key, _ = jr.split(key)
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    acquisition_function_values = acquisition_function(test_X)
    assert acquisition_function_values.shape == (num_test_points, 1)
