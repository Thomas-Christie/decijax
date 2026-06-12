from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
from decijax.test_functions.continuous_functions import NegativeForrester
from decijax.utility_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)
from decijax.utils import OBJECTIVE

from tests.utils import generate_dummy_conjugate_posterior


def test_probability_of_improvement_gives_correct_value_for_a_seed():
    key = jr.key(42)
    neg_forrester = NegativeForrester()
    dataset = neg_forrester.generate_dataset(num_points=10, key=key)
    posterior = generate_dummy_conjugate_posterior(dataset, neg_forrester)
    posteriors = {OBJECTIVE: posterior}
    datasets = {OBJECTIVE: dataset}

    pi_utility_builder = ProbabilityOfImprovement()
    pi_utility = pi_utility_builder.build_utility_function(
        posteriors=posteriors, datasets=datasets, key=key
    )

    test_X = neg_forrester.generate_test_points(num_points=10, key=key)
    utility_values = pi_utility(test_X)

    # Computing the expected utility values
    predictive_dist = posterior(test_X, train_data=dataset)
    predictive_mean = predictive_dist.mean
    predictive_std = predictive_dist.stddev()

    # Computing best_y as the max. of the posterior predictive mean
    # over the training set.
    predictive_dist_for_training_data = posterior(dataset.X, train_data=dataset)
    best_y = predictive_dist_for_training_data.mean.max()

    # 1 - Gaussian CDF computed "by hand"
    x_ = (best_y - predictive_mean) / predictive_std
    expected_utility_values = 1 - 0.5 * (
        1 + jax.scipy.special.erf(x_ / jnp.sqrt(2))
    ).reshape(-1, 1)

    assert utility_values.shape == (10, 1)
    assert jnp.isclose(utility_values, expected_utility_values).all()
