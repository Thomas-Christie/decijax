from dataclasses import dataclass

import jax.numpy as jnp
from gpjax.dataset import Dataset
from gpjax.gps import ConjugatePosterior
from jaxtyping import (
    Array,
    Float,
    Key,
)

from decijax.models.base import (
    SamplePath,
    SupportsGaussianPrediction,
    SupportsSamplePaths,
)
from decijax.models.distributions import GaussianDistribution


@dataclass
class GPJaxConjugateGP(SupportsGaussianPrediction, SupportsSamplePaths):
    """Adapter wrapping a GPJax `ConjugatePosterior` + its training data.

    The dataset is assumed to be in *prediction space*: if the posterior was fit
    on standardised targets, pass the standardised dataset and the standardised
    predictions / incumbent fall out consistently.

    Args:
        posterior: The fitted GPJax posterior.
        dataset: Training data in prediction space.
        num_features: Number of random Fourier features used for pathwise
            (Thompson) sampling.

    Raises:
        ValueError: If `posterior` is not a `ConjugatePosterior`, or if
            `num_features` is not a positive integer.
    """

    posterior: ConjugatePosterior
    dataset: Dataset
    num_features: int = 100

    def __post_init__(self):
        if not isinstance(self.posterior, ConjugatePosterior):
            raise ValueError(
                "GPJaxConjugateGP requires a ConjugatePosterior (i.e. a Gaussian "
                "likelihood). Non-conjugate posteriors are not currently supported, "
                "as decijax acquisitions assume Gaussian predictive distributions."
            )
        if self.num_features <= 0:
            raise ValueError(
                "The number of random Fourier features must be a positive integer."
            )

    @property
    def training_inputs(self) -> Float[Array, "N D"]:
        return self.dataset.X

    @property
    def observations(self) -> Float[Array, "N 1"]:
        return self.dataset.y

    def predict(self, x: Float[Array, "N D"]) -> GaussianDistribution:
        """Marginal Gaussian predictive at ``x``, over the latent function (noise-free).

        Args:
            x: Query points.

        Returns:
            The marginal Gaussian distribution at each query point, with a
            leading sample axis ``S == 1``.
        """
        latent = self.posterior(x, self.dataset)
        # GPJax yields 1-D marginals; lift to the [S, N] contract with S == 1.
        mean = jnp.atleast_2d(latent.mean)
        variance = jnp.atleast_2d(latent.variance)
        return GaussianDistribution(mean, variance)

    def draw_sample_paths(self, num_samples: int, key: Key[Array, ""]) -> SamplePath:
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
            train_data=self.dataset,
            key=key,
            num_features=self.num_features,
        )
