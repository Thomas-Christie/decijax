import jax.numpy as jnp
import jax.random as jr
import pytest
from decijax.models import (
    GPJaxConjugateGP,
    GPJaxConjugateGPBuilder,
)
from decijax.test_functions import NegativeForrester
from gpjax.dataset import Dataset
from gpjax.gps import (
    ExactPosterior,
    Prior,
)
from gpjax.kernels import Matern52
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.objectives import conjugate_mll
from gpjax.parameters import (
    Real,
    val,
)

INITIAL_LENGTHSCALE = jnp.array([0.5])
INITIAL_SIGNAL_VAR = jnp.array(1.0)
INITIAL_MEAN = jnp.array([1.0])


def _make_builder(max_num_optimization_iters=10, **kwargs) -> GPJaxConjugateGPBuilder:
    mean_function = Constant(constant=Real(value=INITIAL_MEAN))
    kernel = Matern52(
        lengthscale=INITIAL_LENGTHSCALE, variance=INITIAL_SIGNAL_VAR, n_dims=1
    )
    prior = Prior(mean_function=mean_function, kernel=kernel)
    return GPJaxConjugateGPBuilder(
        prior=prior,
        likelihood=Gaussian(),
        optimization_objective=conjugate_mll,
        max_num_optimization_iters=max_num_optimization_iters,
        **kwargs,
    )


@pytest.mark.parametrize("max_num_optimization_iters", [0, -1, -10])
def test_builder_erroneous_num_optimization_iterations_raises_error(
    max_num_optimization_iters: int,
):
    with pytest.raises(ValueError):
        _make_builder(max_num_optimization_iters=max_num_optimization_iters)


@pytest.mark.parametrize("num_datapoints", [5, 50])
def test_build_produces_fitted_conjugate_model(num_datapoints: int):
    # Use an identity transform so the fitted dataset matches the input dataset and we
    # can compare training objectives directly.
    builder = _make_builder(observation_transform=lambda y: y)
    test_function = NegativeForrester()
    dataset = test_function.generate_dataset(num_points=num_datapoints, key=jr.key(42))
    non_optimized_posterior = builder.prior * Gaussian()

    model = builder.build(dataset, key=jr.key(42))

    assert isinstance(model, GPJaxConjugateGP)
    assert isinstance(model.posterior, ExactPosterior)
    # Hyperparameters moved away from their initial values. Fitted parameters are
    # still wrapped (e.g. `PositiveReal`), so unwrap them before comparing to arrays.
    prior = model.posterior.prior
    assert val(prior.kernel.lengthscale) != INITIAL_LENGTHSCALE
    assert val(prior.kernel.variance) != INITIAL_SIGNAL_VAR
    assert val(prior.mean_function.constant) != INITIAL_MEAN
    # Optimization increases the marginal log-likelihood
    optimized_model = model.posterior.prior * model.posterior.likelihood
    assert conjugate_mll(optimized_model, dataset) > conjugate_mll(
        non_optimized_posterior, dataset
    )


def test_build_standardizes_observations_by_default():
    builder = _make_builder()
    test_function = NegativeForrester()
    dataset = test_function.generate_dataset(num_points=50, key=jr.key(42))
    # NegativeForrester is already roughly standardised, so shift and scale it to make
    # the raw data unambiguously unstandardised.
    dataset = Dataset(X=dataset.X, y=10.0 * dataset.y + 5.0)

    model = builder.build(dataset, key=jr.key(42))

    # The model carries the standardised observations (zero mean, unit variance).
    assert jnp.allclose(model.observations.mean(), 0.0, atol=1e-6)
    assert jnp.allclose(model.observations.std(), 1.0, atol=1e-6)
    # Inputs are never transformed.
    assert jnp.all(jnp.equal(model.training_inputs, dataset.X))


def test_build_with_identity_transform_leaves_observations_unchanged():
    builder = _make_builder(observation_transform=lambda y: y)
    test_function = NegativeForrester()
    dataset = test_function.generate_dataset(num_points=20, key=jr.key(42))
    # NegativeForrester is already roughly standardised, so shift and scale it to make
    # the raw data unambiguously unstandardised.
    dataset = Dataset(X=dataset.X, y=10.0 * dataset.y + 5.0)

    model = builder.build(dataset, key=jr.key(42))

    assert jnp.all(jnp.equal(model.observations, dataset.y))
    assert jnp.all(jnp.equal(model.training_inputs, dataset.X))
