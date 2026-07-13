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


class ProbabilityOfImprovement(AbstractSinglePointAcquisitionFunctionBuilder):
    r"""
    An acquisition function which returns the probability of improvement
    of the objective function over the best observed value.

    More precisely, given a predictive posterior distribution of the objective
    function $`f`$, the probability of improvement at a test point $`x`$ is defined as:
    $$`\text{PI}(x) = \text{Prob}[f(x) > f(x_{\text{best}})]`$$
    where $`x_{\text{best}}`$ is the maximiser of the posterior mean
    at previously observed values (to handle noisy observations).

    The probability of improvement can be easily computed using the
    cumulative distribution function of the standard normal distribution $`\Phi`$:
    $$`\text{PI}(x) = 1 - \Phi\left(\frac{f(x_{\text{best}}) - \mu}{\sigma}\right)`$$
    where $`\mu`$ and $`\sigma`$ are the mean and standard deviation of the
    predictive distribution of the objective function at $`x`$.

    References
    ----------
    [1] Kushner, H. J. (1964).
    A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise.
    Journal of Basic Engineering, 86(1), 97-106.

    [2] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016).
    Taking the human out of the loop: A review of Bayesian optimization.
    Proceedings of the IEEE, 104(1), 148-175. doi: 10.1109/JPROC.2015.2494218
    """

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: Key[Array, ""],
    ) -> SinglePointAcquisitionFunction:
        """
        Constructs the probability of improvement acquisition function
        using the predictive posterior of the objective function.

        For models carrying a leading sample axis (e.g. fully Bayesian GPs), the
        probability of improvement is computed per sample and averaged.

        Args:
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support Gaussian prediction.
            key (Key[Array, ""]): JAX PRNG key used for random number generation. Since
                the probability of improvement is computed deterministically from the
                predictive posterior, the key is not used.

        Returns:
            SinglePointAcquisitionFunction: the probability of improvement acquisition
                function.
        """
        self.check_objective_present(models)
        objective_model = models[OBJECTIVE]

        if not isinstance(objective_model, SupportsGaussianPrediction):
            raise ValueError(
                "Objective model must support Gaussian prediction to compute the "
                "Probability of Improvement using a Gaussian CDF."
            )

        best_y = get_best_latent_observation_val(objective_model)  # [S, 1]

        def _probability_of_improvement(
            x: Float[Array, "N D"],
        ) -> Float[Array, "N 1"]:
            predictive_dist = objective_model.predict(x)
            mean = predictive_dist.mean  # [S, N]
            std = predictive_dist.stddev  # [S, N]
            pi = 1.0 - norm.cdf((best_y - mean) / std)  # [S, N]
            return jnp.mean(pi, axis=0)[:, None]  # marginalise over S -> [N, 1]

        return _probability_of_improvement
