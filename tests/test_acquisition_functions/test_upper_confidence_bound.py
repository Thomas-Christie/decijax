import jax.numpy as jnp
import jax.random as jr
import pytest
from decijax.acquisition_functions.upper_confidence_bound import UpperConfidenceBound
from decijax.models import GPJaxConjugateGP
from decijax.test_functions.continuous_functions import (
    AbstractContinuousTestFunction,
    NegativeForrester,
    NegativeLogarithmicGoldsteinPrice,
)
from decijax.typing import KeyArray
from decijax.utils import OBJECTIVE

from tests.utils import generate_dummy_conjugate_posterior


@pytest.mark.parametrize("beta", [0.0, 4.0])
@pytest.mark.parametrize(
    "test_target_function",
    [NegativeForrester(), NegativeLogarithmicGoldsteinPrice()],
)
@pytest.mark.parametrize("key", [jr.key(42), jr.key(10)])
def test_upper_confidence_bound_matches_closed_form(
    beta: float,
    test_target_function: AbstractContinuousTestFunction,
    key: KeyArray,
):
    # UCB is closed-form, so it can be checked exactly rather than statistically.
    # beta = 0 is included as the degenerate case: pure exploitation of the mean.
    data_key, acq_key, test_key = jr.split(key, 3)
    dataset = test_target_function.generate_dataset(num_points=10, key=data_key)
    posterior = generate_dummy_conjugate_posterior(dataset, test_target_function)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    ucb_fn = UpperConfidenceBound(beta=beta).build_acquisition_function(
        {OBJECTIVE: model}, acq_key
    )
    test_x = test_target_function.generate_test_points(100, test_key)
    ucb = ucb_fn(test_x)
    latent_dist = posterior.predict(test_x, dataset)
    expected = latent_dist.mean + jnp.sqrt(beta) * jnp.sqrt(latent_dist.variance)

    assert ucb.shape == (100, 1)
    assert jnp.all(jnp.equal(ucb, expected[:, None]))


def test_upper_confidence_bound_negative_beta_raises_error():
    with pytest.raises(ValueError):
        UpperConfidenceBound(beta=-1.0)
