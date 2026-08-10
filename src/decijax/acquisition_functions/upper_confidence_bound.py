"""Upper confidence bound acquisition function."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from decijax.acquisition_functions.base import (
    AbstractSinglePointAcquisitionFunctionBuilder,
    SinglePointAcquisitionFunction,
)
from decijax.models import (
    ProbabilisticModel,
    SupportsGaussianPrediction,
)
from decijax.typing import KeyArray
from decijax.utils import OBJECTIVE


@dataclass
class UpperConfidenceBound(AbstractSinglePointAcquisitionFunctionBuilder):
    r"""Upper Confidence Bound acquisition function.

    Given a predictive posterior distribution of the objective function $f$, the
    upper confidence bound at a test point $x$ is defined as:

    $$\text{UCB}(x) = \mu(x) + \sqrt{\beta}\,\sigma(x)$$

    where $\mu$ and $\sigma$ are the mean and standard deviation of the predictive
    distribution of the objective function at $x$.

    Note that $\beta$ weights the *variance*, following [1], so it is
    $\sqrt{\beta}$ that counts the standard deviations by which the bound sits
    above the mean. [1] also grows $\beta$ with the iteration count to bound
    cumulative regret, but instead we follow BoTorch and keep $\beta$ fixed.

    Attributes:
        beta: Non-negative trade-off between exploitation and exploration. A
            value of $4$, for instance, places the bound two standard deviations
            above the mean.

    References:
    ----------
    [1] Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010).
    Gaussian process optimization in the bandit setting: No regret and experimental design.
    International Conference on Machine Learning (ICML).
    """

    beta: float

    def __post_init__(self):
        """Perform post-initialisation validity checks.

        Raises:
            ValueError: If `beta` is negative, which would leave its square root
                undefined.
        """
        if self.beta < 0.0:
            raise ValueError("Beta must be non-negative.")

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        r"""Build the Upper Confidence Bound acquisition function.

        For models carrying a leading sample axis (e.g. fully Bayesian GPs), the
        upper confidence bound is computed per sample and averaged, which is the
        correct marginalisation
        $`\mathbb{E}_\theta[\alpha_{\text{UCB},\theta}(\mathbf{x})]`$.

        Args:
            models: Dictionary of models used to form the acquisition function. One
                model must correspond to the `OBJECTIVE` key and support Gaussian
                prediction, as we use the objective posterior to form the acquisition
                function.
            key: JAX PRNG key used for random number generation. Since the upper
                confidence bound is computed deterministically, the key is not used.

        Returns:
            The Upper Confidence Bound acquisition function to be *maximised* in
            order to decide which point to query next.
        """
        self.check_objective_present(models)
        objective_model = models[OBJECTIVE]

        if not isinstance(objective_model, SupportsGaussianPrediction):
            raise ValueError(
                "Objective model must support Gaussian prediction to compute the "
                "Upper Confidence Bound."
            )

        sqrt_beta = jnp.sqrt(self.beta)

        def _upper_confidence_bound(x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
            latent_dist = objective_model.predict(x)
            mean = latent_dist.mean  # [S, N]
            std = latent_dist.stddev  # [S, N]
            ucb = mean + sqrt_beta * std  # [S, N]
            return jnp.mean(ucb, axis=0)[:, None]  # marginalise over S -> [N, 1]

        return _upper_confidence_bound
