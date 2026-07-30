"""Expected improvement acquisition functions."""

from collections.abc import Mapping

import jax.numpy as jnp
from jax.nn import logmeanexp
from jax.scipy.special import erfc
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

_LOG_SQRT_2PI = 0.5 * jnp.log(2.0 * jnp.pi)
_LOG_SQRT_PI_DIV_2 = 0.5 * jnp.log(jnp.pi / 2.0)
_NEG_INV_SQRT_2 = -(2.0**-0.5)
_LOG_2 = jnp.log(2.0)

# erfcx ~ (sqrt(pi) x)^-1 sum_k c_k x^-2k, with c_k = (-1)^k (2k-1)!! / 2^k, in
# descending degree for `jnp.polyval`.
_ERFCX_COEFFS = [
    7918.06640625,
    -1055.7421875,
    162.421875,
    -29.53125,
    6.5625,
    -1.875,
    0.75,
    -0.5,
    1.0,
]
# Where that expansion reaches machine precision. Must stay *below* the point at
# which erfc underflows (26.54 in double, 9.19 in single) — see `_erfcx`.
_ERFCX_ASYMPTOTIC_BOUND = {64: 13.7, 32: 6.9}


def _log1mexp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(1 - e^x)`$ for $`x < 0`$ without catastrophic cancellation.

    Which form is accurate depends on how close $`e^x`$ is to one, hence the split
    at $`-\log 2`$. See [Mächler,
    2012](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf).

    Args:
        x (Float[Array, "..."]): Strictly negative array.

    Returns:
        Float[Array, "..."]: $`\log(1 - e^x)`$, elementwise.
    """
    # Each branch is clamped to its own side of the split, so the discarded one
    # cannot poison the gradient of the surviving one.
    return jnp.where(
        x > -_LOG_2,
        jnp.log(-jnp.expm1(jnp.maximum(x, -_LOG_2))),
        jnp.log1p(-jnp.exp(jnp.minimum(x, -_LOG_2))),
    )


def _erfcx(x: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute the scaled complementary error function $`e^{x^2}\text{erfc}(x)`$.

    We need to define our own erfcx function, because `jax.scipy.special.erfcx` has
    a bug where it currently returns zero on $`[26.54, 26.64]`$ (double) and
    $`[9.19, 9.42]`$ (single). Once
    [jax-ml/jax#38607](https://github.com/jax-ml/jax/issues/38607) is fixed, replace
    this function with the native JAX implementation.

    Args:
        x (Float[Array, "..."]): Strictly positive array.

    Returns:
        Float[Array, "..."]: $`e^{x^2}\text{erfc}(x)`$, elementwise.
    """
    dtype = jnp.result_type(x)
    bound = _ERFCX_ASYMPTOTIC_BOUND[64 if jnp.finfo(dtype).bits >= 64 else 32]
    large = x > bound
    # Clamped so the discarded branch can neither overflow nor poison the gradient.
    safe_x = jnp.where(large, jnp.ones_like(x), x)
    direct = jnp.exp(jnp.square(safe_x)) * erfc(safe_x)
    asymptotic = jnp.polyval(
        jnp.asarray(_ERFCX_COEFFS, dtype=dtype), 1.0 / jnp.square(x)
    ) / (x * jnp.sqrt(jnp.pi))
    return jnp.where(large, asymptotic, direct)


def _log_abs_z_cdf_div_pdf(z: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(|z| \Phi(z) / \phi(z))`$ for $`z < 0`$.

    Deep in the tail $`\Phi(z)`$ and $`\phi(z)`$ both underflow to zero even though
    their ratio is perfectly ordinary: at $`z = -40`$ both are around $`10^{-349}`$,
    while $`|z| \Phi(z) / \phi(z)`$ is $`0.999`$. `_erfcx` cancels the shared
    $`e^{-z^2/2}`$ factor analytically, so neither tiny number is ever formed.

    Args:
        z (Float[Array, "..."]): Strictly negative array.

    Returns:
        Float[Array, "..."]: Elementwise; strictly negative, approaching zero from
            below.
    """
    return jnp.log(_erfcx(_NEG_INV_SQRT_2 * z) * jnp.abs(z)) + _LOG_SQRT_PI_DIV_2


def _log_ei_helper(z: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Compute $`\log(\phi(z) + z \Phi(z))`$ in a numerically stable manner.

    Evaluated directly this underflows to zero, gradient included, once
    $`z \lesssim -6`$ in double precision. Three regimes keep it finite ([Ament et
    al., 2023](https://arxiv.org/abs/2310.20708)): the naive form above $`z = -1`$,
    the cancellation-free $`\log \phi(z) + \log(1 - e^w)`$ below it, and that
    term's leading order $`-2 \log|z|`$ in the deep tail. Each branch clamps on
    *both* sides of its boundary to keep gradients clean.

    Args:
        z (Float[Array, "..."]): Scaled improvement.

    Returns:
        Float[Array, "..."]: $`\log(\phi(z) + z \Phi(z))`$, elementwise.
    """
    # Conservative bound from BoTorch, below which `_log1mexp` can no longer resolve w.
    dtype = jnp.result_type(z)
    eps_bound = -1e6 if jnp.finfo(dtype).bits >= 64 else -1e3
    bound = -1.0

    z_upper = jnp.where(z < bound, bound, z)
    log_ei_upper = jnp.log(norm.pdf(z_upper) + z_upper * norm.cdf(z_upper))

    z_lower = jnp.where(z > bound, bound, z)
    z_eps = jnp.where(z_lower < eps_bound, eps_bound, z_lower)
    # w is negative by construction; the clamp only guards it rounding up to zero,
    # which would send `_log1mexp` to the logarithm of a negative number.
    w = jnp.minimum(_log_abs_z_cdf_div_pdf(z_eps), -jnp.finfo(dtype).tiny)
    log_ei_lower = (-0.5 * z_lower**2 - _LOG_SQRT_2PI) + jnp.where(
        z_lower > eps_bound, _log1mexp(w), -2.0 * jnp.log(jnp.abs(z_lower))
    )

    return jnp.where(z > bound, log_ei_upper, log_ei_lower)


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
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support Gaussian prediction, as we use the objective posterior
                to form the acquisition function.
            key (KeyArray): JAX PRNG key used for random number generation. Since
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
    Advances in Neural Information Processing Systems, 36.
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
            models (Mapping[str, ProbabilisticModel]): Dictionary of models used to form
                the acquisition function. One model must correspond to the `OBJECTIVE`
                key and support Gaussian prediction, as we use the objective posterior
                to form the acquisition function.
            key (KeyArray): JAX PRNG key used for random number generation. Since
                the log expected improvement is computed deterministically, the key is
                not used.

        Returns:
            SinglePointAcquisitionFunction: The Log Expected Improvement acquisition
                function to be *maximised* in order to decide which point to query
                next.
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
