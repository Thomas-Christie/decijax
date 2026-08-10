from collections.abc import Callable
from functools import partial

import jax.random as jr
import pytest
from decijax.acquisition_functions.base import (
    AbstractSinglePointAcquisitionFunctionBuilder,
)
from decijax.acquisition_functions.expected_improvement import (
    ExpectedImprovement,
    LogExpectedImprovement,
)
from decijax.acquisition_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)
from decijax.acquisition_functions.thompson_sampling import ThompsonSampling
from decijax.acquisition_functions.upper_confidence_bound import UpperConfidenceBound
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

# Most builders are parametrized as bare classes, but `UpperConfidenceBound` takes a
# required `beta`, so the lists below hold zero-argument factories rather than types.
AcquisitionFunctionBuilderFactory = Callable[
    [], AbstractSinglePointAcquisitionFunctionBuilder
]
UPPER_CONFIDENCE_BOUND = pytest.param(
    partial(UpperConfidenceBound, beta=4.0), id="UpperConfidenceBound"
)


@pytest.mark.filterwarnings(
    "ignore::UserWarning"
)  # Sampling with tfp causes JAX to raise a UserWarning due to some internal logic around jnp.argsort
@pytest.mark.parametrize(
    "acquisition_function_builder",
    [
        ExpectedImprovement,
        LogExpectedImprovement,
        ProbabilityOfImprovement,
        ThompsonSampling,
        UPPER_CONFIDENCE_BOUND,
    ],
)
def test_acquisition_function_no_objective_model_raises_error(
    acquisition_function_builder: AcquisitionFunctionBuilderFactory,
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
    [
        ExpectedImprovement,
        LogExpectedImprovement,
        ProbabilityOfImprovement,
        ThompsonSampling,
        UPPER_CONFIDENCE_BOUND,
    ],
)
def test_model_without_required_capability_raises_error(
    acquisition_function_builder: AcquisitionFunctionBuilderFactory,
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
    [
        ExpectedImprovement,
        LogExpectedImprovement,
        ProbabilityOfImprovement,
        ThompsonSampling,
        UPPER_CONFIDENCE_BOUND,
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
def test_acquisition_functions_have_correct_shapes(
    acquisition_function_builder: AcquisitionFunctionBuilderFactory,
    test_target_function: AbstractContinuousTestFunction,
    num_test_points: int,
    key: KeyArray,
):
    data_key, acq_key, test_key = jr.split(key, 3)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    model = generate_dummy_conjugate_model(dataset)
    models = {OBJECTIVE: model}
    acquisition_builder = acquisition_function_builder()
    acquisition_function = acquisition_builder.build_acquisition_function(
        models, acq_key
    )
    test_X = test_target_function.generate_test_points(num_test_points, test_key)
    acquisition_function_values = acquisition_function(test_X)
    assert acquisition_function_values.shape == (num_test_points, 1)
