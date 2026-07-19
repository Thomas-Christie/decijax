"""Decision makers driving the ask-tell optimisation loop."""

import copy
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from beartype.typing import (
    Callable,
    Dict,
    List,
    Mapping,
)
from gpjax.dataset import Dataset
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)

from decijax.acquisition_functions import (
    AbstractAcquisitionFunctionBuilder,
    ThompsonSampling,
)
from decijax.acquisition_maximizer import AbstractAcquisitionMaximizer
from decijax.models import (
    AbstractModelBuilder,
    ProbabilisticModel,
)
from decijax.search_space import AbstractSearchSpace
from decijax.utils import FunctionEvaluator


@dataclass
class AbstractDecisionMaker(ABC):
    """Abstract base class to handle the core decision making loop.

    The decision making loop is split into two key steps, `ask` and `tell`. The `ask`
    step is typically used to decide which point to query next. The `tell` step is
    typically used to update models and datasets with newly queried points. These steps
    can be combined in a 'run' loop which alternates between asking which point to query
    next and telling the decision maker about the newly queried point having evaluated
    the black-box function of interest at this point.

    Attributes:
        search_space: Search space over which we can evaluate the function(s) of interest.
        model_builders: dictionary of model builders, which are used to (re)fit models
            throughout the decision making loop. These objects are typically referred to
            as `models` in the model-based decision making literature. Tags are used to
            distinguish between models. In a typical Bayesian optimisation setup one of
            the tags will be `OBJECTIVE`, defined in `decision_making.utils`.
        datasets: dictionary of datasets, which are augmented with observations
            throughout the decision making loop. These are the canonical record of
            observations, in their original (untransformed) space, and are used to refit
            the models via the `model_builders`. Tags are used to distinguish datasets,
            and correspond to tags in `model_builders`.
        key: JAX random key, used to generate random numbers.
        batch_size: Number of points to query at each step of the decision making
            loop. Note that `SinglePointAcquisitionFunction`s are only capable of generating
            one point to be queried at each iteration of the decision making loop.
        post_ask: List of functions to be executed after each ask step.
        post_tell: List of functions to be executed after each tell step.
    """

    search_space: AbstractSearchSpace
    model_builders: Dict[str, AbstractModelBuilder]
    datasets: Dict[str, Dataset]
    key: KeyArray
    batch_size: int
    post_ask: List[
        Callable
    ]  # Specific type is List[Callable[[AbstractDecisionMaker, Float[Array, ["B D"]]], None]] but causes Beartype issues
    post_tell: List[
        Callable
    ]  # Specific type is List[Callable[[AbstractDecisionMaker], None]] but causes Beartype issues

    def __post_init__(self):
        """Checks for batch size validity, model/dataset consistency, and builds models.

        At initialisation we check that the model builders and datasets are consistent
        (i.e. have the same tags), and then build the models, fitting them using the
        corresponding datasets.

        Raises:
            ValueError: If `batch_size` is less than 1, or if `model_builders` and
                `datasets` do not have the same tags (keys).
        """
        self.datasets = copy.copy(
            self.datasets
        )  # Ensure initial datasets passed in to DecisionMaker are not mutated from within

        if self.batch_size < 1:
            raise ValueError(
                f"Batch size must be greater than 0, got {self.batch_size}."
            )

        # Check that model builders and datasets are consistent
        if self.model_builders.keys() != self.datasets.keys():
            raise ValueError(
                "Model builders and datasets must have the same keys. "
                f"Got model builders keys {self.model_builders.keys()} and "
                f"datasets keys {self.datasets.keys()}."
            )

        # Build models
        self.models: Dict[str, ProbabilisticModel] = {}
        for tag, model_builder in self.model_builders.items():
            self.models[tag] = model_builder.build(self.datasets[tag], self.key)

    @abstractmethod
    def ask(self, key: KeyArray) -> Float[Array, "B D"]:
        """Get the point(s) to be queried next.

        Args:
            key (KeyArray): JAX PRNG key for controlling random state.

        Returns:
            Float[Array, "1 D"]: Point to be queried next
        """
        raise NotImplementedError

    def tell(self, observation_datasets: Mapping[str, Dataset], key: KeyArray):
        """Add newly observed data to datasets and refit the corresponding models.

        Args:
            observation_datasets: dictionary of datasets containing new observations.
                Tags are used to distinguish datasets, and correspond to tags in
                `model_builders` and `self.datasets`.
            key: JAX PRNG key for controlling random state.
        """
        if observation_datasets.keys() != self.datasets.keys():
            raise ValueError(
                "Observation datasets and existing datasets must have the same keys. "
                f"Got observation datasets keys {observation_datasets.keys()} and "
                f"existing datasets keys {self.datasets.keys()}."
            )

        for tag, observation_dataset in observation_datasets.items():
            self.datasets[tag] += observation_dataset

        for tag, model_builder in self.model_builders.items():
            key, _ = jr.split(key)
            self.models[tag] = model_builder.build(self.datasets[tag], key)

    def run(
        self, n_steps: int, black_box_function_evaluator: FunctionEvaluator
    ) -> Mapping[str, Dataset]:
        """Run the decision making loop continuously for for `n_steps`.

        This is broken down into three main steps:
        1. Call the `ask` method to get the point to be queried next.
        2. Call the `black_box_function_evaluator` to evaluate the black box functions
        of interest at the point chosen to be queried.
        3. Call the `tell` method to update the datasets and posteriors with the newly
        observed data.

        In addition to this, after the `ask` step, the functions in the `post_ask` list
        are executed, taking as arguments the decision maker and the point chosen to be
        queried next. Similarly, after the `tell` step, the functions in the `post_tell`
        list are executed, taking the decision maker as the sole argument.

        Args:
            n_steps (int): Number of steps to run the decision making loop for.
            black_box_function_evaluator (FunctionEvaluator): Function evaluator which
                evaluates the black box functions of interest at supplied points.

        Returns:
            Mapping[str, Dataset]: Dictionary of datasets containing the observations
            made throughout the decision making loop, as well as the initial data
            supplied when initialising the `DecisionMaker`.
        """
        for _ in range(n_steps):
            query_point = self.ask(self.key)

            for post_ask_method in self.post_ask:
                post_ask_method(self, query_point)

            self.key, _ = jr.split(self.key)
            observation_datasets = black_box_function_evaluator(query_point)
            self.tell(observation_datasets, self.key)

            for post_tell_method in self.post_tell:
                post_tell_method(self)

        return self.datasets


@dataclass
class AcquisitionDrivenDecisionMaker(AbstractDecisionMaker):
    """Class which handles the core decision making loop in a model-based setup.

    In this setup we use surrogate model(s) for the function(s) of interest, and define
    an acquisition function which characterises how useful it would be to query a given
    point within the search space given the data we have observed so far. This can then
    be used to decide which point(s) to query next.

    The decision making loop is split into two key steps, `ask` and `tell`. The `ask`
    step forms a `AcquisitionFunction` from the current `posteriors` and `datasets` and
    returns the point which maximises it. It also stores the formed acquisition function
    under the attribute `self.current_acquisition_function` so that it can be called,
    for instance for plotting, after the `ask` function has been called. The `tell` step
    adds a newly queried point to the `datasets` and updates the `posteriors`.

    This can be run as a typical ask-tell loop, or the `run` method can be used to run
    the decision making loop for a fixed number of steps. Moreover, the `run` method
    executes the functions in `post_ask` and `post_tell` after each ask and tell step
    respectively. This enables the user to add custom functionality, such as the ability
    to plot values of interest during the optimization process.

    Attributes:
        acquisition_function_builder (AbstractAcquisitionFunctionBuilder): Object which
                builds acquisition functions from posteriors and datasets, to decide where to query next. In a typical Bayesian optimisation setup the point chosen to be queried next is the point which maximizes the acquisition function.
        acquisition_maximizer (AbstractAcquisitionMaximizer): Object which maximizes
            acquisition functions over the search space.
    """

    acquisition_function_builder: AbstractAcquisitionFunctionBuilder
    acquisition_maximizer: AbstractAcquisitionMaximizer

    def __post_init__(self):
        """Initialise the base class and validate batch size compatibility.

        Raises:
            NotImplementedError: If `batch_size` is greater than 1 with an
                acquisition function builder other than `ThompsonSampling`.
        """
        super().__post_init__()
        if self.batch_size > 1 and not isinstance(
            self.acquisition_function_builder, ThompsonSampling
        ):
            raise NotImplementedError(
                "Batch size > 1 currently only supported for Thompson sampling."
            )

    def ask(self, key: KeyArray) -> Float[Array, "B D"]:
        """Form acquisition function(s) and return the point(s) which maximise them.

        This method also stores the acquisition function(s) in
        `self.current_acquisition_functions` so that they can be accessed after the ask
        function has been called. This is useful for non-deterministic acquisition
        functions, which may differ between calls to `ask` due to the splitting of
        `self.key`.

        Note that in general `SinglePointAcquisitionFunction`s are only capable of
        generating one point to be queried at each iteration of the decision making loop
        (i.e. `self.batch_size` must be 1). However, Thompson sampling can be used in a
        batched setting by drawing a batch of different samples from the GP posterior.
        This is done by calling `build_acquisition_function` with different keys
        sequentilly, and optimising each of these individual samples in sequence in
        order to obtain `self.batch_size` points to query next.

        Args:
            key (KeyArray): JAX PRNG key for controlling random state.

        Returns:
            Float[Array, "B D"]: Point(s) to be queried next.
        """
        self.current_acquisition_functions = []
        maximizers = []
        # We currently only allow Thompson sampling to be run with batch size > 1. More
        # batched acquisition functions may be added in the future.
        if isinstance(self.acquisition_function_builder, ThompsonSampling) or (
            (not isinstance(self.acquisition_function_builder, ThompsonSampling))
            and (self.batch_size == 1)
        ):
            # Draw 'self.batch_size' Thompson samples and optimize each of them in order to
            # obtain 'self.batch_size' points to query next.
            for _ in range(self.batch_size):
                decision_function = (
                    self.acquisition_function_builder.build_acquisition_function(
                        self.models, key
                    )
                )
                self.current_acquisition_functions.append(decision_function)

                _, key = jr.split(key)
                maximizer = self.acquisition_maximizer.maximize(
                    decision_function, self.search_space, key
                )
                maximizers.append(maximizer)
                _, key = jr.split(key)

            maximizers = jnp.concatenate(maximizers)
            return maximizers
        else:
            raise NotImplementedError(
                "Only Thompson sampling currently supports batch size > 1."
            )
