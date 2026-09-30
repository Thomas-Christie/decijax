"""Gaussian process models."""

from dataclasses import dataclass

import jax.numpy as jnp
from gpjax.gps import ExactPosterior
from jaxtyping import (
    Array,
    Float,
)

from decijax.models.base import (
    SamplePath,
    SupportsGaussianPrediction,
    SupportsSamplePaths,
)
from decijax.models.distributions import GaussianDistribution
from decijax.typing import KeyArray


@dataclass(frozen=True, eq=False)
class GPJaxConjugateGP(SupportsGaussianPrediction, SupportsSamplePaths):
    """Adapter wrapping a GPJax `ExactPosterior`.

    The posterior's training data is in *prediction space*: if the model was fit on
    standardised targets, the posterior's dataset contains these standardised targets
    and it will predict in this standardised space.

    Args:
        posterior: The fitted GPJax model conditioned on its training data.
        num_features: Number of random Fourier features used for pathwise
            (Thompson) sampling.

    Raises:
        ValueError: If `posterior` is not an `ExactPosterior`, or if `num_features`
            is not a positive integer.
    """

    posterior: ExactPosterior
    num_features: int = 100

    def __post_init__(self):
        """Validate that the posterior is exact and `num_features` is positive.

        Raises:
            ValueError: If `posterior` is not an `ExactPosterior`, or if
                `num_features` is not a positive integer.
        """
        if not isinstance(self.posterior, ExactPosterior):
            raise ValueError(
                "GPJaxConjugateGP requires an ExactPosterior (i.e. a ConjugateModel "
                "conditioned on data). Non-conjugate models are not currently "
                "supported."
            )
        if self.num_features <= 0:
            raise ValueError(
                "The number of random Fourier features must be a positive integer."
            )

    @property
    def training_inputs(self) -> Float[Array, "N D"]:
        """Inputs the model was conditioned on."""
        return self.posterior.train_data.X

    @property
    def observations(self) -> Float[Array, "N 1"]:
        """Observed targets, in prediction space."""
        return self.posterior.train_data.y

    def predict(self, x: Float[Array, "N D"]) -> GaussianDistribution:
        """Marginal Gaussian predictive at ``x``, over the latent function (noise-free).

        Args:
            x: Query points.

        Returns:
            The marginal Gaussian distribution at each query point, with a leading
            sample axis ``S == 1``.
        """
        latent = self.posterior(x, covariance="diagonal")
        # GPJax yields 1-D marginals; lift to the [S, N] contract with S == 1.
        mean = jnp.atleast_2d(latent.mean)
        variance = jnp.atleast_2d(latent.variance)
        return GaussianDistribution(mean, variance)

    def draw_sample_paths(self, num_samples: int, key: KeyArray) -> SamplePath:
        """Draw differentiable posterior sample paths via decoupled sampling.

        Args:
            num_samples: Number of sample paths to draw.
            key: PRNG key controlling the draw.

        Returns:
            A single callable mapping ``[N, D] -> [N, num_samples]``, evaluable
            and differentiable at arbitrary inputs.
        """
        return self.posterior.sample_approx(
            num_samples=num_samples,
            key=key,
            num_features=self.num_features,
        )
