"""Model builders."""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from typing import TypeAlias

import gpjax as gpx
import jax.numpy as jnp
import paramax
from gpjax.dataset import Dataset
from gpjax.gps import (
    AbstractLikelihood,
    AbstractPrior,
)
from gpjax.objectives import Objective
from jaxtyping import (
    Array,
    Float,
)

from decijax.models.base import ProbabilisticModel
from decijax.models.gps import GPJaxConjugateGP
from decijax.typing import KeyArray

LikelihoodBuilder: TypeAlias = Callable[[int], AbstractLikelihood]
"""Takes the number of datapoints and returns an initialised likelihood. Needed
because GPJax likelihoods bake in ``num_datapoints`` at construction, so the
likelihood must be rebuilt whenever the dataset grows."""

ObservationTransform: TypeAlias = Callable[[Float[Array, "N 1"]], Float[Array, "N 1"]]
"""Maps observations (``y``) to the space the model is fit in. Operates on ``y``
only — inputs (``X``) are deliberately never transformed here, since the search
space and acquisition maximiser operate in original ``X`` coordinates and the
model is queried with those same inputs. Applied at fit time only; predictions
stay in this space and are never mapped back."""


def standardize_observations(
    observations: Float[Array, "N 1"],
) -> Float[Array, "N 1"]:
    """Default transform: standardise ``y`` to zero mean / unit variance.

    Guards against zero variance (e.g. a single initial observation), in which
    case the scale is left at 1 rather than dividing by zero.
    """
    mean = observations.mean()
    std = observations.std()
    std = jnp.where(std == 0.0, 1.0, std)
    return (observations - mean) / std


class AbstractModelBuilder(ABC):
    """Builds a `ProbabilisticModel` from the canonical (original-space) dataset.

    It has two responsibilities:

    1. Fit / refit the surrogate to the data.
    2. Apply any observation transform, so the returned model predicts — *and*
       reports its training data / incumbent — in one consistent space.
    """

    @abstractmethod
    def build(self, dataset: Dataset, key: KeyArray) -> ProbabilisticModel:
        """Fit a fresh model to ``dataset`` and return it."""
        raise NotImplementedError


@dataclass
class GPJaxConjugateGPBuilder(AbstractModelBuilder):
    """GPJax conjugate GP builder.

    Args:
        prior: GP prior.
        likelihood_builder: Builds the likelihood for a given number of
            datapoints (see `LikelihoodBuilder`).
        optimization_objective: Objective *maximised* when fitting
            hyperparameters (e.g. `conjugate_mll`). `gpx.fit_scipy` itself
            always minimises, so `build` negates this internally.
        max_num_optimization_iters: Maximum number of L-BFGS-B iterations run by
            `gpx.fit_scipy`. Must be at least 1. Defaults to 500, matching
            `gpx.fit_scipy`'s own default.
        num_features: Number of random Fourier features used for sample paths.
        observation_transform: Maps observations to the space the model is fit
            in, applied at fit time only (see `ObservationTransform`). Defaults
            to `standardize_observations`.

    Raises:
        ValueError: If `max_num_optimization_iters` is less than 1.
    """

    prior: AbstractPrior
    likelihood_builder: LikelihoodBuilder
    optimization_objective: Objective
    max_num_optimization_iters: int = field(default=500)
    num_features: int = field(default=100)
    observation_transform: ObservationTransform = standardize_observations

    def __post_init__(self):
        """Validate the builder configuration.

        Raises:
            ValueError: If `max_num_optimization_iters` is less than 1.
        """
        if self.max_num_optimization_iters < 1:
            raise ValueError("max_num_optimization_iters must be greater than 0.")

    def build(self, dataset: Dataset, key: KeyArray) -> GPJaxConjugateGP:
        """Fit a fresh conjugate GP to `dataset` and return it.

        Args:
            dataset: Canonical, original-space dataset to fit to.
            key: Unused by this builder (`gpx.fit_scipy` is deterministic).

        Returns:
            A `GPJaxConjugateGP` carrying the transformed dataset it was fit on.
        """
        # X passes through untouched; only observations are transformed.
        fit_dataset = Dataset(X=dataset.X, y=self.observation_transform(dataset.y))
        posterior = self.prior * self.likelihood_builder(fit_dataset.n)
        # gpx.fit_scipy minimises; negate so `optimization_objective` is maximised.
        negated_objective = lambda p, d: -self.optimization_objective(p, d)
        opt_posterior, _ = gpx.fit_scipy(
            model=posterior,
            objective=negated_objective,
            train_data=fit_dataset,
            max_iters=self.max_num_optimization_iters,
            safe=True,
            verbose=False,
        )
        # gpx.fit_scipy returns a model whose parameters are still wrapped
        # (e.g. `NonNegativeReal`, or `NonTrainable` for frozen parameters).
        # `posterior(...)`/`sample_approx(...)` don't recursively unwrap nested
        # wrappers themselves, so a parameter frozen via `paramax.non_trainable`
        # (e.g. a fixed observation noise) would otherwise break prediction.
        # Unwrapping once here resolves every parameter to its plain value.
        # TODO(Thomas-Christie): Come back to this
        opt_posterior = paramax.unwrap(opt_posterior)
        # The model carries the *transformed* dataset, so its predictions and its
        # incumbent (best observed) are automatically consistent.
        return GPJaxConjugateGP(
            posterior=opt_posterior,
            dataset=fit_dataset,
            num_features=self.num_features,
        )
