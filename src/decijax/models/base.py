"""Root model abstract base class and predictive capability mix-ins."""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Callable
from typing import TypeAlias

from jaxtyping import (
    Array,
    Float,
)

from decijax.models.distributions import (
    AbstractGaussianDistribution,
    AbstractMultivariateGaussianDistribution,
)
from decijax.typing import KeyArray

SamplePath: TypeAlias = Callable[[Float[Array, "N D"]], Float[Array, "N S"]]
"""A drawn posterior sample *path*: a callable mapping query points ``[N, D]`` to
``num_samples`` function values ``[N, S]``. The same path is returned consistently
(fixed given its key), which is what Thompson sampling optimises over."""


class ProbabilisticModel(ABC):
    """Root interface for surrogate models.

    Deliberately carries *no* predictive capability of its own — those are mixed
    in via the `Supports*` interfaces below, so a model only implements what the
    acquisitions it targets actually require. The one universal obligation is to
    expose the data the model was conditioned on, in the **same space its
    predictions live in**: a model fit on standardised targets returns
    standardised predictions *and* standardised training inputs / observations.
    """

    @property
    @abstractmethod
    def training_inputs(self) -> Float[Array, "N D"]:
        """Inputs the model was conditioned on, in prediction space."""
        raise NotImplementedError

    @property
    @abstractmethod
    def observations(self) -> Float[Array, "N 1"]:
        """Observed targets, in prediction space."""
        raise NotImplementedError


class SupportsGaussianPrediction(ProbabilisticModel):
    """Capability: closed-form Gaussian marginals (for instance, required by EI)."""

    @abstractmethod
    def predict(self, x: Float[Array, "N D"]) -> AbstractGaussianDistribution:
        """Marginal Gaussian predictive at ``x``."""
        raise NotImplementedError


class SupportsSamplePaths(ProbabilisticModel):
    """Capability: differentiable function-space samples.

    For instance, required by Thompson sampling.
    """

    @abstractmethod
    def draw_sample_paths(self, num_samples: int, key: KeyArray) -> SamplePath:
        """Draw ``num_samples`` differentiable posterior sample *paths*.

        Returns a single callable mapping ``[N, D] -> [N, num_samples]`` (e.g. via
        RFF / decoupled sampling). Distinct from `SupportsPosteriorSamples`: this
        yields a *function* you can evaluate and differentiate at arbitrary inputs
        and re-query consistently, rather than values at a fixed point set.
        """
        raise NotImplementedError


class SupportsJointPrediction(ProbabilisticModel):
    """Capability: joint Gaussian (with covariance).

    For instance, required by analytic qEI.
    """

    @abstractmethod
    def predict_joint(
        self, x: Float[Array, "N D"]
    ) -> AbstractMultivariateGaussianDistribution:
        """Predict the joint distribution over the ``N`` query points.

        Yields the full cross-covariance between the points (not just marginal
        variances).
        """
        raise NotImplementedError


class SupportsPosteriorSamples(ProbabilisticModel):
    """Capability: jointly-correlated samples at a point set. Required by MC / qEI."""

    @abstractmethod
    def draw_samples(
        self, num_samples: int, key: KeyArray, x: Float[Array, "N D"]
    ) -> Float[Array, "S N num_samples"]:
        """Draw ``num_samples`` jointly-correlated samples at the query points.

        For each of the ``S`` posterior samples, returns ``num_samples`` draws
        from the *joint* distribution over the ``N`` query points (column ``j``
        is one joint draw over all points) — not ``N`` independent marginals.
        """
        raise NotImplementedError
