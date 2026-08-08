import jax.numpy as jnp
import jax.random as jr
import pytest
from decijax.acquisition_maximizer import (
    AbstractSinglePointAcquisitionMaximizer,
    ContinuousSinglePointAcquisitionMaximizer,
    _get_top_k_query_points,
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
@pytest.mark.parametrize("k", [1, 5])
def test_top_k_query_points_returns_correct_points(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    k: int,
):
    query_points = test_function.generate_test_points(1000, key=key)
    acquisition_function = test_function.evaluate
    acquisition_vals = acquisition_function(query_points)[:, 0]
    true_top_k_indices = jnp.argsort(-acquisition_vals)[:k]
    true_top_k_points = query_points[true_top_k_indices]
    top_k_points = _get_top_k_query_points(query_points, acquisition_function, k)
    assert top_k_points.shape == (k, dimensionality)
    assert top_k_points.dtype == jnp.float64
    assert jnp.equal(top_k_points, true_top_k_points).all()
    assert jnp.equal(
        acquisition_function(top_k_points)[:, 0],
        acquisition_vals[true_top_k_indices],
    ).all()


@pytest.mark.parametrize("num_initial_samples", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_initial_samples(
    num_initial_samples: int,
):
    with pytest.raises(ValueError):
        ContinuousSinglePointAcquisitionMaximizer(
            num_initial_samples=num_initial_samples, num_optimization_runs=1
        )


@pytest.mark.parametrize("num_optimization_runs", [0, -1, -10])
def test_continuous_maximizer_raises_error_with_erroneous_num_optimization_runs(
    num_optimization_runs: int,
):
    with pytest.raises(ValueError):
        ContinuousSinglePointAcquisitionMaximizer(
            num_initial_samples=1, num_optimization_runs=num_optimization_runs
        )


@pytest.mark.parametrize(
    "num_initial_samples, num_optimization_runs", [(1, 2), (5, 10), (10, 11)]
)
def test_continuous_maximizer_raises_error_with_too_many_optimization_runs(
    num_initial_samples: int,
    num_optimization_runs: int,
):
    with pytest.raises(ValueError):
        ContinuousSinglePointAcquisitionMaximizer(
            num_initial_samples=num_initial_samples,
            num_optimization_runs=num_optimization_runs,
        )


@pytest.mark.parametrize(
    "test_function, dimensionality",
    [(NegativeForrester(), 1), (NegativeLogarithmicGoldsteinPrice(), 2)],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
@pytest.mark.parametrize("num_optimization_runs", [1, 3])
def test_continuous_maximizer_returns_same_point_with_same_key(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    num_optimization_runs: int,
):
    continuous_maximizer_one = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_optimization_runs=num_optimization_runs
    )
    continuous_maximizer_two = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_optimization_runs=num_optimization_runs
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
@pytest.mark.parametrize("num_optimization_runs", [1, 3])
def test_continuous_maximizer_finds_correct_point(
    test_function: AbstractContinuousTestFunction,
    dimensionality: int,
    key: KeyArray,
    num_optimization_runs: int,
):
    continuous_acquisition_maximizer = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_optimization_runs=num_optimization_runs
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
def test_continuous_maximizer_jaxopt_component(key: KeyArray):
    quadratic = NegativeQuadratic()
    continuous_acquisition_maximizer = ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1,  # Force JaxOpt L-GFBS-B to do the heavy lifting
        num_optimization_runs=1,
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
