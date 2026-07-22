"""Probability of Improvement acquisition functions."""

from collections.abc import Mapping

import jax.numpy as jnp
from jax.nn import logmeanexp
from jax.scipy.special import log_ndtr
from jax.scipy.stats import norm
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
from decijax.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)


class ProbabilityOfImprovement(AbstractSinglePointAcquisitionFunctionBuilder):
    r"""Standard Probability of Improvement acquisition function.

    Given a predictive posterior distribution of the objective function $f$, the
    probability of improvement at a test point $x$ is defined as:

    $$\text{PI}(x) = P[f(x) > f(x^*)]$$

    where $x^*$ is the maximiser of the posterior mean at previously observed values
    (to handle noisy observations).

    The probability of improvement can be easily computed using the
    cumulative distribution function of the standard normal distribution $\Phi$:

    $$\text{PI}(x) = \Phi\left(\frac{\mu - f(x^*)}{\sigma}\right)$$

    where $\mu$ and $\sigma$ are the mean and standard deviation of the
    predictive distribution of the objective function at $x$.

    References:
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
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        """Build the Probability of Imprvoement acquisition function.

        For models carrying a leading sample axis (e.g. fully Bayesian GPs), the
        probability of improvement is computed per sample and averaged.

        Args:
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support Gaussian prediction.
            key (KeyArray): JAX PRNG key used for random number generation. Since
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
            pi = norm.cdf((mean - best_y) / std)  # [S, N]
            return jnp.mean(pi, axis=0)[:, None]  # marginalise over S -> [N, 1]

        return _probability_of_improvement


class LogProbabilityOfImprovement(AbstractSinglePointAcquisitionFunctionBuilder):
    r"""Numerically stable Log Probability of Improvement acquisition function.

    Given a predictive posterior distribution of the objective function $f$, the
    log probability of improvement at a test point $x$ is defined as:

    $$\text{LogPI}(x) = \log(P[f(x) > f(x^*)])$$

    where $x^*$ is the maximiser of the posterior mean at previously observed values
    (to handle noisy observations).

    The computation of the probability of improvement in log space enables one to use a
    more numerically stable method for calculating the log of the cumulative
    distribution function of the standard normal distribution, provided by the
    `jax.scipy.special.log_ndtr` function.

    References:
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
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        """Build the Log Probability of Improvement acquisition function.

        For models carrying a leading sample axis (e.g. fully Bayesian GPs), the
        log probability of improvement is computed per sample and averaged.

        Args:
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support Gaussian prediction.
            key (KeyArray): JAX PRNG key used for random number generation. Since
                the log probability of improvement is computed deterministically from
                the predictive posterior, the key is not used.

        Returns:
            SinglePointAcquisitionFunction: the log probability of improvement
                acquisition function.
        """
        self.check_objective_present(models)
        objective_model = models[OBJECTIVE]

        if not isinstance(objective_model, SupportsGaussianPrediction):
            raise ValueError(
                "Objective model must support Gaussian prediction to compute the "
                "Log Probability of Improvement using a Gaussian CDF."
            )

        best_y = get_best_latent_observation_val(objective_model)  # [S, 1]

        def _log_probability_of_improvement(
            x: Float[Array, "N D"],
        ) -> Float[Array, "N 1"]:
            predictive_dist = objective_model.predict(x)
            mean = predictive_dist.mean  # [S, N]
            std = predictive_dist.stddev  # [S, N]
            log_pi = log_ndtr((mean - best_y) / std)  # [S, N]
            return logmeanexp(log_pi, axis=0)[:, None]  # marginalise over S -> [N, 1]

        return _log_probability_of_improvement
