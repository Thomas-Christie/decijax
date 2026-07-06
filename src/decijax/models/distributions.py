from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)


class AbstractGaussianDistribution(ABC):
    r"""Minimal Gaussian predictive interface consumed by analytic acquisition
    functions (EI, PI, UCB).

    Every quantity carries a leading *sample* axis ``S`` so that one interface
    spans:

    - point-estimate models (``S == 1``),
    - fully Bayesian models (``S`` = number of hyperparameter / MCMC samples),
    - ensembles (``S`` = ensemble size).

    Acquisition functions compute their value *per sample* and then reduce over
    ``S`` (e.g. a mean for EI/PI). For expectation-based acquisitions this is the
    correct marginalisation, since ``E_theta[alpha_theta(x)] = mean_s alpha_s(x)``.
    A point-estimate model just has ``S == 1`` and the reduction is a no-op.

    Only *marginals* live here. Cross-point covariance is a separate capability
    (`SupportsJointPrediction`) because single-point acquisitions never need it
    and many models (BNNs, ensembles) cannot supply it cheaply.
    """

    @property
    @abstractmethod
    def mean(self) -> Float[Array, "S N"]:
        """Marginal predictive mean at each query point, per sample."""
        raise NotImplementedError

    @property
    @abstractmethod
    def variance(self) -> Float[Array, "S N"]:
        """Marginal predictive variance at each query point, per sample."""
        raise NotImplementedError

    @property
    def stddev(self) -> Float[Array, "S N"]:
        """Marginal predictive standard deviation at each query point, per sample."""
        return jnp.sqrt(self.variance)


@dataclass(frozen=True)
class GaussianDistribution(AbstractGaussianDistribution):
    """Concrete Gaussian distribution container."""

    _mean: Float[Array, "S N"]
    _variance: Float[Array, "S N"]

    @property
    def mean(self) -> Float[Array, "S N"]:
        return self._mean

    @property
    def variance(self) -> Float[Array, "S N"]:
        return self._variance


class AbstractMultivariateGaussianDistribution(AbstractGaussianDistribution):
    """Joint Gaussian over the query points, adding the cross-point covariance.

    Required by batch / Monte-Carlo acquisitions, which draw *jointly correlated*
    samples across the candidate batch. Kept separate from the marginal interface so
    that supplying a full covariance is opt-in.
    """

    @property
    @abstractmethod
    def covariance(self) -> Float[Array, "S N N"]:
        """Joint covariance across query points, per sample."""
        raise NotImplementedError
