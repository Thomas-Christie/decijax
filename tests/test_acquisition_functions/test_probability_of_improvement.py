from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
from decijax.models import GPJaxConjugateGP
from decijax.test_functions.continuous_functions import NegativeForrester
from decijax.acquisition_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)
from decijax.utils import OBJECTIVE

from tests.utils import generate_dummy_conjugate_posterior


def test_probability_of_improvement_gives_correct_value_for_a_seed():
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset, neg_forrester)
    model = GPJaxConjugateGP(posterior=posterior, dataset=dataset)
    models = {OBJECTIVE: model}

    pi_acquisition_builder = ProbabilityOfImprovement()
    pi_acquisition = pi_acquisition_builder.build_acquisition_function(models, key)

    test_X = neg_forrester.generate_test_points(num_points=10, key=key)
    acquisition_values = pi_acquisition(test_X)

    # Computing the expected acquisition values
    predictive_dist = posterior(test_X, train_data=dataset)
    predictive_mean = predictive_dist.mean
    predictive_std = predictive_dist.stddev()

    # Computing best_y as the max. of the posterior predictive mean
    # over the training set.
    predictive_dist_for_training_data = posterior(dataset.X, train_data=dataset)
    best_y = predictive_dist_for_training_data.mean.max()

    # 1 - Gaussian CDF computed "by hand"
    x_ = (best_y - predictive_mean) / predictive_std
    expected_acquisition_values = 1 - 0.5 * (
        1 + jax.scipy.special.erf(x_ / jnp.sqrt(2))
    ).reshape(-1, 1)

    assert acquisition_values.shape == (10, 1)
    assert jnp.isclose(acquisition_values, expected_acquisition_values).all()
