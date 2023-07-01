"""
Defines fixtures to be shared across many tests.
"""

import numpy as np

import tensorflow_probability as tfp

import pytest


@pytest.fixture
def normal_dist():
    return tfp.distributions.Independent(tfp.distributions.Normal(
        loc=np.linspace(-2.0, 2.0, 5, dtype='float32'),
        scale=1.0,
    ),
                                         reinterpreted_batch_ndims=1)


@pytest.fixture
def vonmises_dist():
    return tfp.distributions.Independent(tfp.distributions.VonMises(
        loc=np.linspace(-3.0, 3.0, 5, dtype='float32'),
        concentration=1.0,
    ),
                                         reinterpreted_batch_ndims=1)


@pytest.fixture
def normal_sample(normal_dist):
    return normal_dist.sample(10)


@pytest.fixture
def vonmises_sample(vonmises_dist):
    return vonmises_dist.sample(10)
