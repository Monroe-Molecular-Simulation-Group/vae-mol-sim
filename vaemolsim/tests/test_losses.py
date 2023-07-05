"""
Tests for losses module.
"""

# import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pytest

from vaemolsim import losses


@pytest.fixture
def get_prob_dists():
    # Try one blockwise and one independent - same effect, but good to test
    # Sets up so the tow distributions are Gaussians with means offset by 1
    # So the KL divergence should be 0.5 for each independent dimension
    # Since KL divergence sums over independent dimensions, the result should be 1
    dist_a = tfp.distributions.Blockwise([
        tfp.distributions.Normal(loc=tf.ones((100, )), scale=tf.ones((100, ))),
        tfp.distributions.Normal(loc=tf.ones((100, )), scale=tf.ones((100, )))
    ])
    dist_b = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros((100, 2)), scale=tf.ones((100, 2))),
                                           reinterpreted_batch_ndims=1)
    return dist_a, dist_b


def test_logprobloss(normal_dist, normal_sample):
    lp = losses.LogProbLoss()
    loss = lp(normal_sample, normal_dist)
    assert loss.shape == tuple()
    assert loss == -tf.reduce_mean(normal_dist.log_prob(normal_sample))
    lp_nored = losses.LogProbLoss(reduction='none')
    loss_nored = lp_nored(normal_sample, normal_dist)
    assert loss_nored.shape == (normal_sample.shape[0], )


def test_inforegularizer(get_prob_dists):
    ir = losses.InfoRegularizer()
    assert ir.sample_dist == 'dist_a'
    with pytest.raises(NotImplementedError, match='In any subclass'):
        ir(*get_prob_dists)
    ir_b = losses.InfoRegularizer(sample_dist='dist_b')
    assert ir_b.sample_dist == 'dist_b'
    with pytest.raises(ValueError, match='sample_dist'):
        losses.InfoRegularizer(sample_dist='dist_c')


def test_nonregularizer(get_prob_dists):
    nr = losses.NonRegularizer()
    assert nr(*get_prob_dists) == 0.0


def test_kldivergenceestimate(get_prob_dists):
    kl = losses.KLDivergenceEstimate()
    out = kl(*get_prob_dists)
    assert out.shape == tuple()
    s = get_prob_dists[0].sample()
    out_with_samples = kl(get_prob_dists[0], get_prob_dists[1], samples=s)
    assert out_with_samples.shape == tuple()
    kl_w = losses.KLDivergenceEstimate(weight=100.0)
    out_weighted = kl_w(get_prob_dists[0], get_prob_dists[1], samples=s)
    assert out_weighted == 100.0 * out_with_samples
    # np.testing.assert_allclose(out, 1.0, atol=0.2)
    # Also test with deterministic distribution
    det_vals = tf.random.uniform(s.shape, -5.0, 5.0)
    det_dist = tfp.distributions.Independent(tfp.distributions.Deterministic(det_vals), reinterpreted_batch_ndims=1)
    out_det = kl(det_dist, get_prob_dists[1])
    assert out_det == -tf.reduce_mean(get_prob_dists[1].log_prob(det_vals))


def test_logprobregularizer(get_prob_dists):
    lpr = losses.LogProbRegularizer()
    s = get_prob_dists[0].sample()
    out = lpr(get_prob_dists[0], get_prob_dists[1], samples=s)
    assert out.shape == tuple()
    vm_dist = tfp.distributions.Independent(tfp.distributions.VonMises(loc=tf.zeros((100, 2)),
                                                                       concentration=tf.ones((100, 2))),
                                            reinterpreted_batch_ndims=1)
    out_other_dist = lpr(vm_dist, get_prob_dists[1], samples=s)
    assert out_other_dist == out
    out_other_dist_sample = lpr(vm_dist, get_prob_dists[1])
    assert out_other_dist_sample != out


def test_reversekldivergenceestimate(get_prob_dists):
    rkl = losses.ReverseKLDivergenceEstimate()
    assert rkl.sample_dist == 'dist_b'
    s = get_prob_dists[0].sample()
    out = rkl(get_prob_dists[0], get_prob_dists[1], samples=s)
    assert out.shape == tuple()
    kl = losses.KLDivergenceEstimate()
    out_kl = kl(get_prob_dists[1], get_prob_dists[0], samples=s)
    assert out == out_kl
