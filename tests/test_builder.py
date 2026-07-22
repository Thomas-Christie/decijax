import jax.numpy as jnp
import jax.random as jr
import pytest
from decijax.models import (
    GPJaxConjugateGP,
    GPJaxConjugateGPBuilder,
)
from decijax.test_functions import NegativeForrester
from gpjax.gps import (
    ConjugatePosterior,
    Prior,
)
from gpjax.kernels import Matern52
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.objectives import conjugate_mll
from gpjax.parameters import Real


def gaussian_likelihood_builder(num_datapoints: int) -> Gaussian:
    return Gaussian(num_datapoints=num_datapoints)


def _make_builder(max_num_optimization_iters=10, **kwargs) -> GPJaxConjugateGPBuilder:
    mean_function = Constant(constant=Real(value=jnp.array([1.0])))
    kernel = Matern52(lengthscale=jnp.array([0.5]), variance=jnp.array(1.0), n_dims=1)
    prior = Prior(mean_function=mean_function, kernel=kernel)
    return GPJaxConjugateGPBuilder(
        prior=prior,
        likelihood_builder=gaussian_likelihood_builder,
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
    non_optimized_posterior = builder.prior * gaussian_likelihood_builder(
        num_datapoints
    )

    model = builder.build(dataset, key=jr.key(42))

    assert isinstance(model, GPJaxConjugateGP)
    assert isinstance(model.posterior, ConjugatePosterior)
    assert model.posterior.likelihood.num_datapoints == num_datapoints
    # Hyperparameters moved away from their initial values
    assert model.posterior.prior.kernel.lengthscale != jnp.array([0.5])
    assert model.posterior.prior.kernel.variance != jnp.array(1.0)
    assert model.posterior.prior.mean_function.constant != jnp.array([1.0])
    # Optimization increases the marginal log-likelihood
    assert conjugate_mll(model.posterior, dataset) > conjugate_mll(
        non_optimized_posterior, dataset
    )


def test_build_standardizes_observations_by_default():
    builder = _make_builder()
    test_function = NegativeForrester()
    dataset = test_function.generate_dataset(num_points=50, key=jr.key(42))

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

    model = builder.build(dataset, key=jr.key(42))

    assert jnp.all(jnp.equal(model.observations, dataset.y))
    assert jnp.all(jnp.equal(model.training_inputs, dataset.X))
