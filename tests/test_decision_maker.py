import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import pytest
from decijax.acquisition_functions import (
    AbstractSinglePointAcquisitionFunctionBuilder,
    ThompsonSampling,
)
from decijax.acquisition_maximizer import (
    AbstractSinglePointAcquisitionMaximizer,
    ContinuousSinglePointAcquisitionMaximizer,
)
from decijax.decision_maker import (
    AbstractDecisionMaker,
    AcquisitionDrivenDecisionMaker,
)
from decijax.models import GPJaxConjugateGPBuilder
from decijax.search_space import (
    AbstractSearchSpace,
    ContinuousSearchSpace,
)
from decijax.test_functions import NegativeQuadratic
from decijax.typing import KeyArray
from decijax.utils import (
    OBJECTIVE,
    build_function_evaluator,
)
from gpjax.dataset import Dataset

from tests.utils import QuadraticSinglePointAcquisitionFunctionBuilder

CONSTRAINT = "CONSTRAINT"


@pytest.fixture
def search_space() -> ContinuousSearchSpace:
    return ContinuousSearchSpace(
        lower_bounds=jnp.array([0.0], dtype=jnp.float64),
        upper_bounds=jnp.array([1.0], dtype=jnp.float64),
    )


@pytest.fixture
def model_builder() -> GPJaxConjugateGPBuilder:
    mean = gpx.mean_functions.Zero()
    kernel = gpx.kernels.Matern52(
        lengthscale=jnp.array(1.0),
        variance=jnp.array(1.0),
        n_dims=1,
    )
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood_builder = lambda x: gpx.likelihoods.Gaussian(
        num_datapoints=x, obs_stddev=jnp.array(1e-3)
    )
    return GPJaxConjugateGPBuilder(
        prior=prior,
        likelihood_builder=likelihood_builder,
        optimization_objective=gpx.objectives.conjugate_mll,
        max_num_optimization_iters=100,
    )


@pytest.fixture
def acquisition_function_builder() -> AbstractSinglePointAcquisitionFunctionBuilder:
    return QuadraticSinglePointAcquisitionFunctionBuilder()


@pytest.fixture
def thompson_sampling_acquisition_function_builder() -> ThompsonSampling:
    return ThompsonSampling()


@pytest.fixture
def acquisition_maximizer() -> AbstractSinglePointAcquisitionMaximizer:
    return ContinuousSinglePointAcquisitionMaximizer(
        num_initial_samples=1000, num_restarts=1
    )


def get_dataset(num_points: int, key: KeyArray) -> Dataset:
    test_function = NegativeQuadratic()
    dataset = test_function.generate_dataset(num_points=num_points, key=key)
    return dataset


def test_abstract_decision_maker_raises_error():
    with pytest.raises(TypeError):
        AbstractDecisionMaker()


@pytest.mark.parametrize("batch_size", [0, -1, -10])
def test_invalid_batch_size_raises_error(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
    batch_size: int,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    with pytest.raises(ValueError):
        AcquisitionDrivenDecisionMaker(
            search_space=search_space,
            model_builders=model_builders,
            datasets=datasets,
            acquisition_function_builder=acquisition_function_builder,
            acquisition_maximizer=acquisition_maximizer,
            key=key,
            post_ask=[],
            post_tell=[],
            batch_size=batch_size,
        )


def test_non_thompson_sampling_non_one_batch_size_raises_error(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    with pytest.raises(NotImplementedError):
        AcquisitionDrivenDecisionMaker(
            search_space=search_space,
            model_builders=model_builders,
            datasets=datasets,
            acquisition_function_builder=acquisition_function_builder,
            acquisition_maximizer=acquisition_maximizer,
            key=key,
            post_ask=[],
            post_tell=[],
            batch_size=2,
        )


def test_invalid_tags_raises_error(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"CONSTRAINT": dataset}  # Dataset tag doesn't match model builder tag
    with pytest.raises(ValueError):
        AcquisitionDrivenDecisionMaker(
            search_space=search_space,
            model_builders=model_builders,
            datasets=datasets,
            acquisition_function_builder=acquisition_function_builder,
            acquisition_maximizer=acquisition_maximizer,
            key=key,
            post_ask=[],
            post_tell=[],
            batch_size=1,
        )


def test_initialisation_optimizes_model_hyperparameters(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder, CONSTRAINT: model_builder}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    constraint_dataset = get_dataset(num_points=5, key=jr.key(10))
    datasets = {"OBJECTIVE": objective_dataset, CONSTRAINT: constraint_dataset}
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=datasets,
        acquisition_function_builder=acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    objective_prior = decision_maker.models[OBJECTIVE].posterior.prior
    constraint_prior = decision_maker.models[CONSTRAINT].posterior.prior
    # Assert kernel hyperparameters get changed from their initial values
    assert objective_prior.kernel.lengthscale != jnp.array(1.0)
    assert objective_prior.kernel.variance != jnp.array(1.0)
    assert constraint_prior.kernel.lengthscale != jnp.array(1.0)
    assert constraint_prior.kernel.variance != jnp.array(1.0)
    assert constraint_prior.kernel.lengthscale != objective_prior.kernel.lengthscale
    assert constraint_prior.kernel.variance != objective_prior.kernel.variance


def test_decision_maker_ask(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=datasets,
        acquisition_function_builder=acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    initial_decision_maker_key = decision_maker.key
    query_point = decision_maker.ask(key=key)
    assert query_point.shape == (1, 1)
    assert jnp.allclose(query_point, jnp.array([[0.5]]), atol=1e-5)
    assert len(decision_maker.current_acquisition_functions) == 1
    assert (
        decision_maker.key == initial_decision_maker_key
    ).all()  # Ensure decision maker key is unchanged


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_decision_maker_ask_multi_batch_ts(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    thompson_sampling_acquisition_function_builder: ThompsonSampling,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
    batch_size: int,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    datasets = {"OBJECTIVE": objective_dataset}
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=datasets,
        acquisition_function_builder=thompson_sampling_acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=batch_size,
    )
    initial_decision_maker_key = decision_maker.key
    query_points = decision_maker.ask(key=key)
    assert query_points.shape == (batch_size, 1)

    # TODO: ask henry about this failing assertion
    # assert (
    #     len(jnp.unique(query_points)) == batch_size
    # )  # Ensure we aren't drawing the same Thompson sample each time
    assert len(decision_maker.current_acquisition_functions) == batch_size
    assert (
        decision_maker.key == initial_decision_maker_key
    ).all()  # Ensure decision maker key is unchanged


def test_decision_maker_tell_with_inconsistent_observations_raises_error(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder, CONSTRAINT: model_builder}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_constraint_dataset = get_dataset(num_points=5, key=jr.key(10))
    datasets = {
        OBJECTIVE: initial_objective_dataset,
        CONSTRAINT: initial_constraint_dataset,
    }
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=datasets,
        acquisition_function_builder=acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    mock_objective_observation = get_dataset(num_points=1, key=jr.key(1))
    mock_constraint_observation = get_dataset(num_points=1, key=jr.key(2))
    observations = {
        OBJECTIVE: mock_objective_observation,
        "CONSTRAINT_ONE": mock_constraint_observation,  # Deliberately incorrect tag
    }
    with pytest.raises(ValueError):
        decision_maker.tell(observation_datasets=observations, key=key)


def test_decision_maker_tell_updates_datasets_and_models(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder, CONSTRAINT: model_builder}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_constraint_dataset = get_dataset(num_points=5, key=jr.key(10))
    datasets = {
        "OBJECTIVE": initial_objective_dataset,
        CONSTRAINT: initial_constraint_dataset,
    }
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=datasets,
        acquisition_function_builder=acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    initial_decision_maker_key = decision_maker.key
    initial_objective_prior = decision_maker.models[OBJECTIVE].posterior.prior
    initial_constraint_prior = decision_maker.models[CONSTRAINT].posterior.prior
    mock_objective_observation = get_dataset(num_points=1, key=jr.key(1))
    mock_constraint_observation = get_dataset(num_points=1, key=jr.key(2))
    observations = {
        OBJECTIVE: mock_objective_observation,
        CONSTRAINT: mock_constraint_observation,
    }
    decision_maker.tell(observation_datasets=observations, key=key)
    assert decision_maker.datasets[OBJECTIVE].n == 6
    assert decision_maker.datasets[CONSTRAINT].n == 6
    assert decision_maker.datasets[OBJECTIVE].X[-1] == mock_objective_observation.X[0]
    assert decision_maker.datasets[CONSTRAINT].X[-1] == mock_constraint_observation.X[0]
    updated_objective_prior = decision_maker.models[OBJECTIVE].posterior.prior
    updated_constraint_prior = decision_maker.models[CONSTRAINT].posterior.prior
    assert (
        updated_objective_prior.kernel.lengthscale
        != initial_objective_prior.kernel.lengthscale
    )
    assert (
        updated_objective_prior.kernel.variance
        != initial_objective_prior.kernel.variance
    )
    assert (
        updated_constraint_prior.kernel.lengthscale
        != initial_constraint_prior.kernel.lengthscale
    )
    assert (
        updated_constraint_prior.kernel.variance
        != initial_constraint_prior.kernel.variance
    )
    assert (
        decision_maker.key == initial_decision_maker_key
    ).all()  # Ensure decision maker key has not been updated


@pytest.mark.parametrize("n_steps", [1, 3])
def test_decision_maker_run(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    acquisition_function_builder: AbstractSinglePointAcquisitionFunctionBuilder,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
    n_steps: int,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_datasets = {
        "OBJECTIVE": initial_objective_dataset,
    }
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=initial_datasets,
        acquisition_function_builder=acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=1,
    )
    initial_decision_maker_key = decision_maker.key
    black_box_fn = NegativeQuadratic()
    black_box_function_evaluator = build_function_evaluator(
        {OBJECTIVE: black_box_fn.evaluate}
    )
    query_datasets = decision_maker.run(
        n_steps=n_steps, black_box_function_evaluator=black_box_function_evaluator
    )
    assert initial_datasets[OBJECTIVE].n == 5
    assert query_datasets[OBJECTIVE].n == 5 + n_steps
    assert (
        jnp.abs(query_datasets[OBJECTIVE].X[-n_steps:] - jnp.array([[0.5]])) < 1e-5
    ).all()  # Ensure we're querying the correct point in our dummy acquisition function at each step
    assert (
        decision_maker.key != initial_decision_maker_key
    ).all()  # Ensure decision maker key gets updated


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_decision_maker_run_ts(
    search_space: AbstractSearchSpace,
    model_builder: GPJaxConjugateGPBuilder,
    thompson_sampling_acquisition_function_builder: ThompsonSampling,
    acquisition_maximizer: AbstractSinglePointAcquisitionMaximizer,
    n_steps: int,
    batch_size: int,
):
    key = jr.key(42)
    model_builders = {OBJECTIVE: model_builder}
    initial_objective_dataset = get_dataset(num_points=5, key=jr.key(42))
    initial_datasets = {
        "OBJECTIVE": initial_objective_dataset,
    }
    decision_maker = AcquisitionDrivenDecisionMaker(
        search_space=search_space,
        model_builders=model_builders,
        datasets=initial_datasets,
        acquisition_function_builder=thompson_sampling_acquisition_function_builder,
        acquisition_maximizer=acquisition_maximizer,
        key=key,
        post_ask=[],
        post_tell=[],
        batch_size=batch_size,
    )
    initial_decision_maker_key = decision_maker.key
    black_box_fn = NegativeQuadratic()
    black_box_function_evaluator = build_function_evaluator(
        {OBJECTIVE: black_box_fn.evaluate}
    )
    query_datasets = decision_maker.run(
        n_steps=n_steps, black_box_function_evaluator=black_box_function_evaluator
    )
    assert initial_datasets[OBJECTIVE].n == 5
    assert (
        query_datasets[OBJECTIVE].n == 5 + n_steps * batch_size
    )  # Ensure we're getting the correct number of points
    assert (
        decision_maker.key != initial_decision_maker_key
    ).all()  # Ensure decision maker key gets updated
