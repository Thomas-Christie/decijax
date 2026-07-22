import jax.numpy as jnp
import jax.random as jr
import pytest
from decijax.acquisition_maximizer import (
    AbstractSinglePointAcquisitionMaximizer,
    ContinuousSinglePointAcquisitionMaximizer,
    _get_discrete_maximizer,
)
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
    NegativeQuadratic,
)
from decijax.typing import KeyArray


def test_abstract_single_batch_acquisition_maximizer():
    with pytest.raises(TypeError):
        AbstractSinglePointAcquisitionMaximizer()


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(NegativeForrester(), 1), (NegativeLogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_discrete_maximizer_returns_correct_point(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
):
    query_points = test_function.generate_test_points(1000, key=key)
    acquisition_function = test_function.evaluate
    acquisition_vals = acquisition_function(query_points)
    true_max_acquisition_val = jnp.max(acquisition_vals)
    discrete_maximizer = _get_discrete_maximizer(query_points, acquisition_function)
    assert discrete_maximizer.shape == (1, dimensionality)
    assert discrete_maximizer.dtype == jnp.float64
    assert acquisition_function(discrete_maximizer)[0][0] == true_max_acquisition_val


@pytest.mark.parametrize("num_initial_samples", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_initial_samples(
    num_initial_samples: int,
):
    with pytest.raises(ValueError):
        ContinuousSinglePointAcquisitionMaximizer(
            num_initial_samples=num_initial_samples, num_restarts=1
        )


@pytest.mark.parametrize("num_restarts", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_restarts(
    num_restarts: int,
):
    with pytest.raises(ValueError):
        ContinuousSinglePointAcquisitionMaximizer(
            num_initial_samples=1, num_restarts=num_restarts
        )


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(NegativeForrester(), 1), (NegativeLogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.parametrize("num_restarts", [1, 3])
def test_continuous_maximizer_returns_same_point_with_same_key(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    num_restarts: int,
):
    continuous_maximizer_one = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=num_restarts
    )
    continuous_maximizer_two = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=num_restarts
    )
    acquisition_function = test_function.evaluate
    maximizer_one = continuous_maximizer_one.maximize(
        acquisition_function=acquisition_function,
        search_space=test_function.search_space,
        key=key,
    )
    maximizer_two = continuous_maximizer_two.maximize(
        acquisition_function=acquisition_function,
        search_space=test_function.search_space,
        key=key,
    )
    assert maximizer_one.shape == (1, dimensionality)
    assert maximizer_one.dtype == jnp.float64
    assert maximizer_two.shape == (1, dimensionality)
    assert maximizer_two.dtype == jnp.float64
    assert jnp.equal(maximizer_one, maximizer_two).all()


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [
        (NegativeForrester(), 1),
        (NegativeLogarithmicGoldsteinPrice(), 2),
    ],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.parametrize("num_restarts", [1, 3])
def test_continuous_maximizer_finds_correct_point(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    num_restarts: int,
):
    continuous_acquisition_maximizer = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=num_restarts
    )
    acquisition_function = test_function.evaluate
    true_acquisition_maximizer = test_function.maximizer
    maximizer = continuous_acquisition_maximizer.maximize(
        acquisition_function=acquisition_function,
        search_space=test_function.search_space,
        key=key,
    )
    assert maximizer.shape == (1, dimensionality)
    assert maximizer.dtype == jnp.float64
    assert jnp.allclose(maximizer, true_acquisition_maximizer, atol=1e-6).all()


@pytest.mark.parametrize("key", [jr.key(42), jr.key(10), jr.key(1)])
@pytest.mark.parametrize("num_restarts", [1, 3])
def test_continuous_maximizer_jaxopt_component(key: KeyArray, num_restarts: int):
    quadratic = NegativeQuadratic()
    continuous_acquisition_maximizer = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1,  # Force JaxOpt L-GFBS-B to do the heavy lifting
        num_restarts=num_restarts,
    )
    acquisition_function = quadratic.evaluate
    true_acquisition_maximizer = quadratic.maximizer
    maximizer = continuous_acquisition_maximizer.maximize(
        acquisition_function=acquisition_function,
        search_space=quadratic.search_space,
        key=key,
    )
    assert maximizer.shape == (1, 1)
    assert maximizer.dtype == jnp.float64
    assert jnp.allclose(maximizer, true_acquisition_maximizer, atol=1e-6).all()
