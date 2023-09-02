"""
Tests for dists module.
"""

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import pytest

from vaemolsim import dists, flows


def test_make_param_transform():
    # Default setup
    pt = dists.make_param_transform()
    assert pt == tf.identity
    assert pt(1.0) == 1.0
    # Given Normal distribution
    pt_norm = dists.make_param_transform(tfp.distributions.Normal)
    out_norm = pt_norm(np.array([0.0, 0.0]))
    assert isinstance(out_norm, dict)
    np.testing.assert_allclose(out_norm['loc'], 0.0, rtol=1e-06)
    np.testing.assert_allclose(out_norm['scale'], tf.math.softplus(0.0), rtol=1e-06)
    # Given von Mises distribution
    pt_vm = dists.make_param_transform(tfp.distributions.VonMises)
    out_vm = pt_vm(np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(out_vm['loc'], np.pi, rtol=1e-06)
    np.testing.assert_allclose(out_vm['concentration'], tf.math.softplus(0.0), rtol=1e-06)


class TestIndependentBlockwise:

    dist_class = dists.IndependentBlockwise

    def test_only_normal(self):
        ib = self.dist_class(2, tfp.distributions.Normal)
        assert len(ib.dist_classes) == 2
        for d in ib.dist_classes:
            assert d == tfp.distributions.Normal
        assert len(ib.param_nums) == 2
        for pn in ib.param_nums:
            assert pn == 2
        inputs = tf.random.normal(np.hstack([10, ib.params_size()]))
        dist = ib(inputs)
        sample = dist.sample(3)  # Check sampling
        _ = dist.log_prob(sample)  # Check log-probabilities
        if self.dist_class == dists.IndependentBlockwise:
            assert isinstance(dist, tfp.distributions.Blockwise)
            for d in dist.distributions:
                assert isinstance(d, tfp.distributions.Normal)
        elif self.dist_class == dists.AutoregressiveBlockwise:
            assert isinstance(dist, tfp.distributions.Autoregressive)
            for d in dist.distribution_fn(dist.sample0).distributions:
                assert isinstance(d, tfp.distributions.Normal)

    def test_mixed(self):
        c_list = [tfp.distributions.VonMises, tfp.distributions.Normal, tfp.distributions.VonMises]
        ib = self.dist_class(3, c_list)
        assert len(ib.dist_classes) == 3
        assert len(ib.param_nums) == 3
        inputs = tf.random.normal(np.hstack([10, ib.params_size()]))
        dist = ib(inputs)
        sample = dist.sample(3)
        _ = dist.log_prob(sample)
        if self.dist_class == dists.IndependentBlockwise:
            assert isinstance(dist, tfp.distributions.Blockwise)
            for d, c in zip(dist.distributions, c_list):
                assert isinstance(d, c)
        elif self.dist_class == dists.AutoregressiveBlockwise:
            assert isinstance(dist, tfp.distributions.Autoregressive)
            for d, c in zip(dist.distribution_fn(dist.sample0).distributions, c_list):
                assert isinstance(d, c)


class TestAutogregressiveBlockwise(TestIndependentBlockwise):

    dist_class = dists.AutoregressiveBlockwise

    def test_cond_input(self):
        cond_input = tf.random.normal([10, 1])
        c_list = [tfp.distributions.VonMises, tfp.distributions.Normal, tfp.distributions.VonMises]
        ib = self.dist_class(3, c_list, conditional=True, conditional_event_shape=cond_input.shape[-1])
        inputs = tf.random.normal(np.hstack([10, ib.params_size()]))
        dist = ib(inputs, conditional_input=cond_input)
        with pytest.raises(ValueError, match='conditional_input'):
            _ = ib(inputs)
        sample = dist.sample(3)
        _ = dist.log_prob(sample)
        assert isinstance(dist, tfp.distributions.Autoregressive)
        for d, c in zip(dist.distribution_fn(dist.sample0).distributions, c_list):
            assert isinstance(d, c)


class TestFlowedDistribution:

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_standard_tfp(self, f_class):
        l_dist = tfp.layers.IndependentNormal(2)
        input_data = tf.random.normal(np.hstack([10, l_dist.params_size(l_dist._event_shape)]))
        l_sample = l_dist(input_data).sample()
        f = f_class()
        _ = f(l_sample)  # Must flow sample from latent through first to build correctly
        fd = dists.FlowedDistribution(f, l_dist)
        assert fd.params_size() == input_data.shape[-1]
        dist = fd(input_data)
        assert isinstance(dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(dist, "bijector")  # Make sure it has a transform

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_independent_blockwise(self, f_class):
        l_dist = dists.IndependentBlockwise(2, tfp.distributions.Normal)
        input_data = tf.random.normal(np.hstack([10, l_dist.params_size()]))
        l_sample = l_dist(input_data).sample()  # Will need sample to build flow correctly
        f = f_class()
        _ = f(l_sample)
        fd = dists.FlowedDistribution(f, l_dist)
        assert fd.params_size() == input_data.shape[-1]
        dist = fd(input_data)
        assert isinstance(dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(dist, "bijector")  # Make sure it has a transform

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_autoregressive_blockwise(self, f_class):
        l_dist = dists.AutoregressiveBlockwise(2, tfp.distributions.Normal)
        input_data = tf.random.normal(np.hstack([10, l_dist.params_size()]))
        l_sample = l_dist(input_data).sample()  # Will need sample to build flow correctly
        f = f_class()
        _ = f(l_sample)
        fd = dists.FlowedDistribution(f, l_dist)
        assert fd.params_size() == input_data.shape[1:]
        dist = fd(input_data)
        assert isinstance(dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(dist, "bijector")  # Make sure it has a transform

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_static_dist(self, f_class):
        l_dist = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Blockwise([tfp.distributions.Normal(
                loc=0.0, scale=1.0)] * 2 + [tfp.distributions.VonMises(loc=0.0, concentration=1.0)], ))
        l_sample = l_dist(None).sample()
        f = f_class()
        _ = f(l_sample)
        fd = dists.FlowedDistribution(f, l_dist)
        dist = fd(None)
        assert isinstance(dist, tfp.distributions.Distribution)
        assert hasattr(dist, "bijector")

    def test_cond_input(self):
        # Just test with independent
        # Doesn't make sense to have autoregressive distribution AND autoregressive flow
        # Technically perfectly fine, but seems unnecessary
        # Also, if try and make both conditional, will fail for autoregressive distribution
        # (because conditional input only passed to flow, not when making distribution)
        # So in theory, an autoregressive distribution will work, but should not be conditional
        l_dist = dists.IndependentBlockwise(2, tfp.distributions.Normal)
        input_data = tf.random.normal(np.hstack([10, l_dist.params_size()]))
        cond_data = tf.random.normal(np.hstack([10, 1]))
        l_sample = l_dist(input_data).sample()  # Will need sample to build flow correctly
        f = flows.RQSSplineMAF(rqs_params={'conditional': True, 'conditional_event_shape': cond_data.shape[-1]}, )
        _ = f(l_sample, conditional_input=cond_data)
        fd = dists.FlowedDistribution(f, l_dist)
        assert fd.params_size() == input_data.shape[-1]
        with pytest.raises(ValueError, match='conditional_input'):
            _ = fd(input_data).sample()
        dist = fd(input_data, conditional_input=cond_data)
        assert isinstance(dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(dist, "bijector")  # Make sure it has a transform

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_batch_norm(self, f_class):
        l_dist = dists.IndependentBlockwise(2, tfp.distributions.Normal)
        input_data = tf.random.normal(np.hstack([10, l_dist.params_size()]))
        l_sample = l_dist(input_data).sample()  # Will need sample to build flow correctly
        f = f_class(batch_norm=True)
        _ = f(l_sample)
        fd = dists.FlowedDistribution(f, l_dist)
        _ = fd(input_data)
        train_bool = []
        for bij in fd.flow.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                train_bool.append(bij.training)
        assert not np.any(train_bool)
        _ = fd(input_data, training=True)
        train_bool = []
        for bij in fd.flow.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                train_bool.append(bij.training)
        assert np.all(train_bool)


class TestStaticFlowedDistribution:

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_basic(self, f_class, normal_dist, normal_sample):
        input_data = tf.random.normal(np.hstack([10, normal_sample.shape[-1]]))
        f = f_class()
        _ = f(normal_sample)  # Must flow sample from latent through first to build correctly
        fd = dists.StaticFlowedDistribution(f, normal_dist)
        dist = fd(input_data)
        assert isinstance(dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(dist, "bijector")  # Make sure it has a transform

    @pytest.mark.parametrize("f_class", [flows.RQSSplineRealNVP, flows.RQSSplineMAF])
    def test_batch_norm(self, f_class, normal_dist, normal_sample):
        input_data = tf.random.normal(np.hstack([10, normal_sample.shape[-1]]))
        f = f_class()
        _ = f(normal_sample)  # Must flow sample from latent through first to build correctly
        fd = dists.StaticFlowedDistribution(f, normal_dist)
        dist = fd(input_data)
        assert isinstance(dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(dist, "bijector")  # Make sure it has a transform
        train_bool = []
        for bij in fd.flow.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                train_bool.append(bij.training)
        assert not np.any(train_bool)
        _ = fd(input_data, training=True)
        train_bool = []
        for bij in fd.flow.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                train_bool.append(bij.training)
        assert np.all(train_bool)


def test_independentvonmises():
    assert dists.IndependentVonMises.params_size(2) == 6
    ivm = dists.IndependentVonMises(2)
    assert ivm.params_size(ivm._event_shape) == 6
    input_data = tf.random.normal(np.hstack([10, ivm.params_size(ivm._event_shape)]))
    dist = ivm(input_data)
    sample = dist.sample()
    assert sample.shape == (input_data.shape[0], ivm._event_shape)
    lp = dist.log_prob(sample)
    assert lp.shape == (input_data.shape[0], )


def test_independentdeterministic():
    assert dists.IndependentDeterministic.params_size(2) == 2
    idm = dists.IndependentDeterministic(2)
    assert idm.params_size(idm._event_shape) == 2
    input_data = tf.random.normal(np.hstack([10, idm.params_size(idm._event_shape)]))
    dist = idm(input_data)
    sample = dist.sample()
    assert sample.shape == (input_data.shape[0], idm._event_shape)
    lp = dist.log_prob(sample)
    assert lp.shape == (input_data.shape[0], )
    # Make sure it's deterministic
    np.testing.assert_allclose(sample, input_data)
