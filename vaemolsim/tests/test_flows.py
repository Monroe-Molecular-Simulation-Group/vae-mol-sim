"""
Tests for the flows module.
"""

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import pytest

from vaemolsim import flows


def test_domain_transform():
    doms = [[0, 20], [-5, 5], [0, np.pi], [0, 1]]
    target = [0, 1]
    x = [np.linspace(a, b, 10) for a, b in doms]
    x = np.vstack(x).T

    # Test moving to target
    bij_to = flows.make_domain_transform(doms, target)
    y_to = bij_to(x)
    assert y_to.shape == x.shape
    assert np.all(np.min(y_to, axis=0) == 0.0)
    assert np.all(np.max(y_to, axis=0) == 1.0)

    # Test moving from target
    bij_from = flows.make_domain_transform(doms, target, from_target=True)
    y_from = bij_from(y_to)
    assert np.allclose(y_from, x)


class TestSplineBijector:

    # Ignore shape specification for input data here
    # Typically, will use in RealNVP architecture, so that
    # input shape will not match output (inputs are the DOFs
    # that do not get transformed)
    def input_data(self, dims):
        return tf.random.normal([10, 3])

    def sb_class(self, dim, **kwargs):
        return flows.SplineBijector(dim, **kwargs)

    def test_default_creation_1d(self):
        sb = self.sb_class(1)
        _ = sb(self.input_data([10, 1]))  # Provide input to get built
        assert sb.data_dim == 1
        assert sb.bin_widths.weights[0].shape == (sb.hidden_dim, sb.data_dim * sb.num_bins)
        assert sb.bin_heights.weights[0].shape == (sb.hidden_dim, sb.data_dim * sb.num_bins)
        assert sb.knot_slopes.weights[0].shape == (sb.hidden_dim, sb.data_dim * (sb.num_bins - 1))

    def test_default_creation_multid(self):
        sb = self.sb_class(5)
        _ = sb(self.input_data([10, 5]))  # Provide input to build
        assert sb.data_dim == 5
        assert sb.bin_widths.weights[0].shape == (sb.hidden_dim, sb.data_dim * sb.num_bins)
        assert sb.bin_heights.weights[0].shape == (sb.hidden_dim, sb.data_dim * sb.num_bins)
        assert sb.knot_slopes.weights[0].shape == (sb.hidden_dim, sb.data_dim * (sb.num_bins - 1))

    def test_custom_creation(self):
        kwargs = {
            'bin_range': [-np.pi, np.pi],
            'num_bins': 20,
            'hidden_dim': 100,
        }
        sb = self.sb_class(5, **kwargs)
        _ = sb(self.input_data([10, 5]))  # Make sure it can be built
        assert sb.bin_min == -np.pi
        assert sb.bin_max == np.pi
        assert sb.num_bins == 20
        assert sb.hidden_dim == 100

    def test_flow_norm_1d(self, normal_sample):
        sb = self.sb_class(1)
        bi = sb(self.input_data([10, 1]))
        t_sample = bi.forward(normal_sample[:, :1])
        assert t_sample.shape == normal_sample[:, :1].shape
        assert not np.all(t_sample == normal_sample[:, :1])

    def test_flow_norm_multid(self, normal_sample):
        sb = self.sb_class(normal_sample.shape[-1])
        bi = sb(self.input_data(normal_sample.shape))
        t_sample = bi.forward(normal_sample)
        assert t_sample.shape == normal_sample.shape
        assert not np.all(t_sample == normal_sample)

    def test_flow_vonmises_1d(self, vonmises_sample):
        sb = self.sb_class(1, bin_range=[-np.pi, np.pi])
        bi = sb(self.input_data([10, 1]))
        t_sample = bi.forward(vonmises_sample[:, :1])
        assert t_sample.shape == vonmises_sample[:, :1].shape
        assert not np.all(t_sample == vonmises_sample[:, :1])

    def test_flow_vonmises_multid(self, vonmises_sample):
        sb = self.sb_class(vonmises_sample.shape[-1], bin_range=[-np.pi, np.pi])
        bi = sb(self.input_data(vonmises_sample.shape))
        t_sample = bi.forward(vonmises_sample)
        assert t_sample.shape == vonmises_sample.shape
        assert not np.all(t_sample == vonmises_sample)


class TestMaskedSplineBijector(TestSplineBijector):

    # Here, input data needs to be same shape as sample data
    # All data gets transformed, but order of dependencies
    # ensured by masking.
    def input_data(self, dims):
        return tf.random.normal(dims)

    def sb_class(self, dim, **kwargs):
        # For masked version, ignore dim argument...
        # For this it MUST be same as input dimension and enforce that during building
        # So to be able to inherit above class, just ignore dim argument here
        return flows.MaskedSplineBijector(**kwargs)

    # Overwrite next two tests... shape of networks is very different
    def test_default_creation_1d(self):
        sb = self.sb_class(1)
        _ = sb(self.input_data([10, 1]))  # Provide input to get built
        assert sb.data_dim == 1
        assert sb.bin_widths.weights[0].shape == (sb.data_dim, sb.hidden_dim)
        assert sb.bin_widths.event_shape == [
            sb.data_dim,
        ]
        assert sb.bin_heights.weights[0].shape == (sb.data_dim, sb.hidden_dim)
        assert sb.bin_heights.event_shape == [
            sb.data_dim,
        ]
        assert sb.knot_slopes.weights[0].shape == (sb.data_dim, sb.hidden_dim)
        assert sb.knot_slopes.event_shape == [
            sb.data_dim,
        ]

    def test_default_creation_multid(self):
        sb = self.sb_class(5)
        _ = sb(self.input_data([10, 5]))  # Provide input to build
        assert sb.data_dim == 5
        assert sb.bin_widths.weights[0].shape == (sb.data_dim, sb.hidden_dim)
        assert sb.bin_widths.event_shape == [
            sb.data_dim,
        ]
        assert sb.bin_heights.weights[0].shape == (sb.data_dim, sb.hidden_dim)
        assert sb.bin_heights.event_shape == [
            sb.data_dim,
        ]
        assert sb.knot_slopes.weights[0].shape == (sb.data_dim, sb.hidden_dim)
        assert sb.knot_slopes.event_shape == [
            sb.data_dim,
        ]

    def test_cond_inputs(self, normal_sample):
        cond_data = tf.random.normal([normal_sample.shape[0], 2])
        sb = self.sb_class(normal_sample.shape[-1], conditional=True, conditional_event_shape=cond_data.shape[-1])
        bi = sb(normal_sample, conditional_input=self.input_data([normal_sample.shape[0], 2]))
        t_sample = bi.forward(normal_sample)
        assert sb.bin_widths._conditional
        assert sb.bin_heights._conditional
        assert sb.knot_slopes._conditional
        assert t_sample.shape == normal_sample.shape
        assert not np.all(t_sample == normal_sample)


class TestRQSSplineRealNVP:

    flow_class = flows.RQSSplineRealNVP

    def test_default_creation(self, normal_sample):
        f = self.flow_class()
        no_train_sample = f(normal_sample)
        train_sample = f(normal_sample, training=True)
        assert len(f.chain.bijectors) == f.num_blocks
        assert no_train_sample.shape == normal_sample.shape
        assert not np.all(no_train_sample == normal_sample)
        np.testing.assert_array_equal(train_sample, no_train_sample)

    def test_custom_creation(self, normal_sample):
        f = self.flow_class(num_blocks=4, batch_norm=True)
        no_train_sample = f(normal_sample)
        train_bool = []
        for bij in f.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                train_bool.append(bij.training)
        assert not np.any(train_bool)
        train_sample = f(normal_sample, training=True)
        train_bool = []
        for bij in f.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                train_bool.append(bij.training)
        del train_sample
        assert np.all(train_bool)
        assert f.num_blocks == 4
        assert len(f.chain.bijectors) == (2 * f.num_blocks - 1)
        assert no_train_sample.shape == normal_sample.shape
        assert not np.all(no_train_sample == normal_sample)

    def test_normal_dist_transform(self, normal_dist, normal_sample):
        f = self.flow_class()
        # To correctly build, still need to pass sample through flow
        # After that, will work for distribution as well
        t_sample = f(normal_sample)
        t_dist = f(normal_dist)
        _ = t_dist.log_prob(t_sample)  # Check log-probability calc
        new_sample = t_dist.sample(10)  # Make sure new transformed distribution can sample
        _ = t_dist.log_prob(new_sample)  # And calculate a log probability
        assert isinstance(t_dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(t_dist, "bijector")  # Make sure it has a transform

    def test_vonmises_dist_transform(self, vonmises_dist, vonmises_sample):
        f = self.flow_class(rqs_params={'bin_range': [-np.pi, np.pi]})
        t_sample = f(vonmises_sample)
        t_dist = f(vonmises_dist)
        _ = t_dist.log_prob(t_sample)  # Check log-probability calc
        new_sample = t_dist.sample(10)  # Make sure new transformed distribution can sample
        _ = t_dist.log_prob(new_sample)  # And calculate a log-probability
        assert isinstance(t_dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(t_dist, "bijector")  # Make sure it has a transform

    def test_before_after_transforms(self, vonmises_dist, vonmises_sample, normal_sample):
        # Test just with identity transformations before and after
        f = self.flow_class(
            rqs_params={'bin_range': [-np.pi, np.pi]},
            before_flow_transform=tfp.bijectors.Identity(),
            after_flow_transform=tfp.bijectors.Identity(),
        )
        t_sample = f(vonmises_sample)
        assert len(f.chain.bijectors) == f.num_blocks + 2
        assert t_sample.shape == vonmises_sample.shape
        # Now test with transformation from one space to another and back to another
        b_trans = flows.make_domain_transform([[-10.0, 10.0]] * normal_sample.shape[-1], [-np.pi, np.pi])
        a_trans = flows.make_domain_transform([[0.0, 1.0]] * normal_sample.shape[-1], [-np.pi, np.pi],
                                              from_target=True)
        f_ba = self.flow_class(
            rqs_params={'bin_range': [-np.pi, np.pi]},
            before_flow_transform=b_trans,
            after_flow_transform=a_trans,
        )
        t_sample_ba = f_ba(normal_sample)
        assert 0.0 <= np.min(t_sample_ba)
        assert 1.0 >= np.max(t_sample_ba)


class TestRQSSplineMAF(TestRQSSplineRealNVP):

    flow_class = flows.RQSSplineMAF

    def test_cond_inputs(self, normal_dist, normal_sample):
        cond_data = tf.random.normal([normal_sample.shape[0], 2])
        f = self.flow_class(rqs_params={'conditional': True, 'conditional_event_shape': 2}, )
        t_sample = f(normal_sample, conditional_input=cond_data)
        with pytest.raises(ValueError, match='conditional_input'):
            _ = f(normal_sample)
        cond_bool = []
        for bij in f.chain.bijectors:
            if isinstance(bij, flows.MaskedSplineBijector):
                cond_bool.append(bij.conditional)
        assert np.all(cond_bool)
        assert t_sample.shape == normal_sample.shape
        assert not np.all(t_sample == normal_sample)
        t_dist = f(normal_dist, conditional_input=cond_data)
        new_sample = t_dist.sample(10)  # Make sure new transformed distribution can sample
        _ = t_dist.log_prob(new_sample)  # And calculate a log probability
        assert isinstance(t_dist, tfp.distributions.Distribution)  # Make sure it's a distribution
        assert hasattr(t_dist, "bijector")  # Make sure it has a transform
