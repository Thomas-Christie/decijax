import jax.numpy as jnp
from beartype.typing import Mapping
from jax.scipy.stats import norm
from jaxtyping import (
    Array,
    Float,
    Key,
)

from decijax.acquisition_functions.base import (
    AbstractSinglePointAcquisitionFunctionBuilder,
    SinglePointAcquisitionFunction,
)
from decijax.models import (
    ProbabilisticModel,
    SupportsGaussianPrediction,
)
from decijax.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)


class ExpectedImprovement(AbstractSinglePointAcquisitionFunctionBuilder):
    """
    Expected Improvement acquisition function as introduced by [Močkus,
    1974](https://link.springer.com/chapter/10.1007/3-540-07165-2_55). The "best"
    incumbent value is defined as the highest posterior mean value evaluated at the
    previously observed points. This enables the acquisition function to be utilised with noisy observations.
    """

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: Key[Array, ""],
    ) -> SinglePointAcquisitionFunction:
        r"""
        Build the Expected Improvement acquisition function. This computes the expected
        improvement over the "best" of the previously observed points, utilising the
        posterior distribution of the surrogate model. For posterior distribution
        $`f(\cdot)`$, and best incumbent value $`\eta`$, this is defined
        as:
        ```math
        \alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}\left[\max(0, f(\mathbf{x}) - \eta)\right]
        ```

        For models carrying a leading sample axis (e.g. fully Bayesian GPs), the
        expected improvement is computed per sample and averaged, which is the correct
        marginalisation $`\mathbb{E}_\theta[\alpha_{\text{EI},\theta}(\mathbf{x})]`$.

        Args:
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support Gaussian prediction, as we use the objective posterior
                to form the acquisition function.
            key (Key[Array, ""]): JAX PRNG key used for random number generation. Since
                the expected improvement is computed deterministically, the key is not
                used.

        Returns:
            SinglePointAcquisitionFunction: The Expected Improvement acquisition
                function to to be *maximised* in order to decide which point to query
                next.
        """
        self.check_objective_present(models)
        objective_model = models[OBJECTIVE]

        if not isinstance(objective_model, SupportsGaussianPrediction):
            raise ValueError(
                "Objective model must support Gaussian prediction to compute the "
                "Expected Improvement."
            )

        eta = get_best_latent_observation_val(objective_model)  # [S, 1]

        def _expected_improvement(x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
            latent_dist = objective_model.predict(x)
            mean = latent_dist.mean  # [S, N]
            std = latent_dist.stddev  # [S, N]
            z = (mean - eta) / std
            # Canonical EI: (mu - eta) * Phi(z) + sigma * phi(z), per sample [S, N].
            ei = (mean - eta) * norm.cdf(z) + std * norm.pdf(z)
            return jnp.mean(ei, axis=0)[:, None]  # marginalise over S -> [N, 1]

        return _expected_improvement
