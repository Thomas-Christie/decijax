from typing import TypeAlias

import jax.numpy as jnp
from beartype.typing import (
    Callable,
    Dict,
    Final,
)
from gpjax.dataset import Dataset
from gpjax.typing import (
    Array,
    Float,
)

from decijax.models import SupportsGaussianPrediction

OBJECTIVE: Final[str] = "OBJECTIVE"
"""
Tag for the objective dataset/function in standard acquisition functions.
"""


FunctionEvaluator: TypeAlias = Callable[[Float[Array, "N D"]], Dict[str, Dataset]]
"""
Type alias for function evaluators, which take an array of points of shape $[N, D]$
and evaluate a set of functions at each point, returning a mapping from function tags
to datasets of the evaluated points. This is the same as the `Observer` in Trieste:
https://github.com/secondmind-labs/trieste/blob/develop/trieste/observer.py
"""


def build_function_evaluator(
    functions: Dict[str, Callable[[Float[Array, "N D"]], Float[Array, "N 1"]]],
) -> FunctionEvaluator:
    """
    Takes a dictionary of functions and returns a `FunctionEvaluator` which can be
    used to evaluate each of the functions at a supplied set of points and return a
    dictionary of datasets storing the evaluated points.
    """
    return lambda x: {tag: Dataset(x, f(x)) for tag, f in functions.items()}


def get_best_latent_observation_val(
    model: SupportsGaussianPrediction,
) -> Float[Array, "S 1"]:
    """
    Returns the best (latent) incumbent value for each posterior sample, defined as
    the maximum of the predictive mean evaluated at the model's observed inputs. In
    the noiseless case this corresponds to the maximum observed value.

    The leading sample axis ``S`` is preserved (``S == 1`` for point-estimate
    models, ``S`` = number of hyperparameter samples for fully Bayesian models), so
    expectation-based acquisitions can compute their incumbent per sample before
    marginalising.
    """
    latent_at_observed = model.predict(model.training_inputs).mean  # [S, N]
    return jnp.max(latent_at_observed, axis=-1, keepdims=True)  # [S, 1]
