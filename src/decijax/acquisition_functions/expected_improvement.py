"""Expected improvement acquisition functions."""

from collections.abc import Mapping

import jax.numpy as jnp
from jax.nn import logmeanexp
from jax.scipy.stats import norm
from jaxtyping import (
    Array,
    Float,
)

from decijax.acquisition_functions.base import (
    AbstractSinglePointAcquisitionFunctionBuilder,
    SinglePointAcquisitionFunction,
)
from decijax.maths import (
    _log1mexp,
    _log_abs_z_cdf_div_pdf,
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


def _log_ei_helper(z: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(\phi(z) + z \Phi(z))`$ in a numerically stable manner.

    Evaluated directly this underflows to zero, gradient included, once
    $`z \lesssim -6`$ in double precision. Implements the piecewise formulation of
    [Ament et al., 2023](https://arxiv.org/abs/2310.20708), Eq. 9:

    ```math
    \begin{cases}
    \log\left(\phi(z) + z \Phi(z)\right) & z > -1 \\
    -\frac{z^2}{2} - c_1 + \text{log1mexp}\left(
        \log\left(\text{erfcx}\left(\frac{-z}{\sqrt{2}}\right)|z|\right) + c_2
    \right) & \tau < z \leq -1 \\
    -\frac{z^2}{2} - c_1 - 2\log|z| & z \leq \tau
    \end{cases}
    ```

    where $`c_1 = \frac{\log(2\pi)}{2}`$ and $`c_2 = \frac{\log(\pi / 2)}{2}`$. The
    threshold $`\tau`$ between the last two branches is $`-10^6`$ in double
    precision and $`-10^3`$ in single, following BoTorch. Each branch clamps on
    *both* sides of its boundaries to keep gradients clean.

    Args:
        z: Scaled improvement.

    Returns:
        $`\log(\phi(z) + z \Phi(z))`$, elementwise.
    """
    if z.dtype == jnp.float64:
        eps_bound = -1e6
    elif z.dtype == jnp.float32:
        eps_bound = -1e3
    else:
        raise NotImplementedError(
            f"LogExpectedImprovement does not support dtype {z.dtype}."
        )

    log_sqrt_2pi = 0.5 * jnp.log(2.0 * jnp.pi)

    # Branch 1 (z > -1): direct computation.
    z_upper = jnp.maximum(z, -1.0)
    direct = jnp.log(norm.pdf(z_upper) + z_upper * norm.cdf(z_upper))

    # Branch 2 (eps_bound < z <= -1): stable computation using erfcx.
    z_mid = jnp.clip(z, eps_bound, -1.0)
    w = _log_abs_z_cdf_div_pdf(z_mid)
    stable = -0.5 * z_mid**2 - log_sqrt_2pi + _log1mexp(w)

    # Branch 3 (z <= eps_bound): asymptotic, where `_log1mexp` can no longer resolve w.
    z_lower = jnp.minimum(z, eps_bound)
    asymptotic = -0.5 * z_lower**2 - log_sqrt_2pi - 2.0 * jnp.log(-z_lower)

    return jnp.where(
        z > -1.0,
        direct,
        jnp.where(z > eps_bound, stable, asymptotic),
    )


class ExpectedImprovement(AbstractSinglePointAcquisitionFunctionBuilder):
    """Standard Expected Improvement acquisition function.

    As introduced by
    [Močkus, 1974](https://link.springer.com/chapter/10.1007/3-540-07165-2_55). The
    "best" incumbent value is defined as the highest posterior mean value evaluated
    at the previously observed points. This enables the acquisition function to be
    utilised with noisy observations.
    """

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        r"""Build the Expected Improvement acquisition function.

        This computes the expected improvement over the "best" of the previously
        observed points, utilising the posterior distribution of the surrogate model.
        For posterior distribution $`f(\cdot)`$, and best incumbent value $`\eta`$,
        this is defined as:

        ```math
        \alpha_{\text{EI}}(\mathbf{x})
        = \mathbb{E}\left[\max(0, f(\mathbf{x}) - \eta)\right]
        ```

        For models carrying a leading sample axis (e.g. fully Bayesian GPs), the
        expected improvement is computed per sample and averaged, which is the correct
        marginalisation $`\mathbb{E}_\theta[\alpha_{\text{EI},\theta}(\mathbf{x})]`$.

        Args:
            models: Dictionary of models used to form the acquisition function. One
                model must correspond to the `OBJECTIVE` key and support Gaussian
                prediction, as we use the objective posterior to form the acquisition
                function.
            key: JAX PRNG key used for random number generation. Since the expected
                improvement is computed deterministically, the key is not used.

        Returns:
            The Expected Improvement acquisition function to be *maximised* in order to
            decide which point to query next.
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


class LogExpectedImprovement(AbstractSinglePointAcquisitionFunctionBuilder):
    r"""Numerically stable Log Expected Improvement acquisition function [1].

    Given a predictive posterior distribution of the objective function $f$, the log
    expected improvement at a test point $x$ is defined as:

    $$\text{LogEI}(x) = \log \mathbb{E}\left[\max(0, f(x) - f(x^*))\right]$$

    where $x^*$ is the maximiser of the posterior mean at previously observed values
    (to handle noisy observations).

    Being a strictly increasing transform of the expected improvement, this shares
    its maximiser exactly, but is far better behaved as an optimisation target:
    expected improvement vanishes to *exactly* zero, gradient included, once the
    scaled improvement falls below roughly $-40$ in double precision, and those flat
    regions come to dominate the search space in higher dimensions.

    References:
    ----------
    [1] Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2023).
    Unexpected improvement to expected improvement for Bayesian optimization.
    Neural Information Processing Systems (NeurIPS).
    """

    def build_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        key: KeyArray,
    ) -> SinglePointAcquisitionFunction:
        r"""Build the Log Expected Improvement acquisition function.

        The expected improvement factorises as $`\sigma \cdot h(z)`$, for scaled
        improvement $`z = \frac{\mu - \eta}{\sigma}`$ and
        $`h(z) = \phi(z) + z\Phi(z)`$, so that:

        ```math
        \alpha_{\text{LogEI}}(\mathbf{x}) = \log \sigma(\mathbf{x}) + \log h(z)
        ```

        with the second term computed by `_log_ei_helper`. For models carrying a
        leading sample axis (e.g. fully Bayesian GPs), it is computed per sample and
        reduced with a log-mean-exp, the correct marginalisation
        $`\log \mathbb{E}_\theta[\alpha_{\text{EI},\theta}(\mathbf{x})]`$.

        Args:
            models: Dictionary of models used to form the acquisition function. One
                model must correspond to the `OBJECTIVE` key and support Gaussian
                prediction, as we use the objective posterior to form the acquisition
                function.
            key: JAX PRNG key used for random number generation. Since the log
                expected improvement is computed deterministically, the key is not
                used.

        Returns:
            The Log Expected Improvement acquisition function to be *maximised* in
            order to decide which point to query next.
        """
        self.check_objective_present(models)
        objective_model = models[OBJECTIVE]

        if not isinstance(objective_model, SupportsGaussianPrediction):
            raise ValueError(
                "Objective model must support Gaussian prediction to compute the "
                "Log Expected Improvement."
            )

        eta = get_best_latent_observation_val(objective_model)  # [S, 1]

        def _log_expected_improvement(x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
            latent_dist = objective_model.predict(x)
            mean = latent_dist.mean  # [S, N]
            std = latent_dist.stddev  # [S, N]
            z = (mean - eta) / std
            log_ei = _log_ei_helper(z) + jnp.log(std)  # [S, N]
            return logmeanexp(log_ei, axis=0)[:, None]  # marginalise over S -> [N, 1]

        return _log_expected_improvement
