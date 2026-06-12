import numpyro.distributions as dist
from beartype.typing import Mapping
from gpjax.dataset import Dataset
from gpjax.gps import ConjugatePosterior
from gpjax.typing import (
    Array,
    KeyArray,
)
from jaxtyping import Num

from decijax.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from decijax.utils import (
    OBJECTIVE,
    get_best_latent_observation_val,
)


class ProbabilityOfImprovement(AbstractSinglePointUtilityFunctionBuilder):
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

    def build_utility_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        """
        Constructs the probability of improvement utility function
        using the predictive posterior of the objective function.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function. One of the posteriors must correspond
            to the `OBJECTIVE` key, as we sample from the objective posterior to form
            the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function. Keys in `datasets` should correspond to
            keys in `posteriors`. One of the datasets must correspond
            to the `OBJECTIVE` key.
            key (KeyArray): JAX PRNG key used for random number generation. Since
            the probability of improvement is computed deterministically
            from the predictive posterior, the key is not used.

        Returns:
            SinglePointUtilityFunction: the probability of improvement utility function.
        """
        self.check_objective_present(posteriors, datasets)

        objective_posterior = posteriors[OBJECTIVE]
        if not isinstance(objective_posterior, ConjugatePosterior):
            raise ValueError(
                "Objective posterior must be a ConjugatePosterior to compute the Probability of Improvement using a Gaussian CDF."
            )

        objective_dataset = datasets[OBJECTIVE]
        if (
            objective_dataset.X is None
            or objective_dataset.n == 0
            or objective_dataset.y is None
        ):
            raise ValueError(
                "Objective dataset must be non-empty to compute the "
                "Probability of Improvement (since we need a "
                "`best_y` value)."
            )

        def probability_of_improvement(x_test: Num[Array, "N D"]):
            best_y = get_best_latent_observation_val(
                objective_posterior, objective_dataset
            )
            predictive_dist = objective_posterior.predict(x_test, objective_dataset)

            normal_dist = dist.Normal(
                loc=predictive_dist.mean,
                scale=predictive_dist.stddev(),
            )

            return (1 - normal_dist.cdf(best_y)).reshape(-1, 1)

        return probability_of_improvement
