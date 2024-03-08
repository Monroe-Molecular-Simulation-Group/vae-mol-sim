"""
Tests for models module.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pytest

from vaemolsim import flows, dists, mappings, losses, models


class TestFlowModel:

    target_dist = tfp.distributions.Independent(tfp.distributions.Uniform(low=[tf.cast(-np.pi, tf.float32)] * 3,
                                                                          high=[tf.cast(np.pi, tf.float32)] * 3),
                                                reinterpreted_batch_ndims=1)
    target_sample = target_dist.sample(10)

    input_data = tf.random.normal([10, 5])

    @pytest.mark.parametrize("f", [
        flows.RQSSplineRealNVP(),
        flows.RQSSplineRealNVP(batch_norm=True),
        flows.RQSSplineMAF(),
        flows.RQSSplineMAF(batch_norm=True),
        flows.RQSSplineMAF(rqs_params={
            'conditional': True,
            'conditional_event_shape': input_data.shape[-1]
        }),
    ])
    def test_static(self, f):
        static_dist = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Blockwise(
            [tfp.distributions.Normal(loc=tf.zeros((tf.shape(t)[0], )), scale=tf.ones((tf.shape(t)[0], )))] * 2 +
            [tfp.distributions.VonMises(loc=tf.zeros((tf.shape(t)[0], )), concentration=tf.ones(
                (tf.shape(t)[0], )))], ))
        if f.conditional:
            _ = f(self.target_sample, conditional_input=self.input_data)
        else:
            _ = f(self.target_sample)
        m = models.FlowModel(f, static_dist)
        assert m.mapping is None
        _ = m(self.target_sample)  # Make sure can pass through model
        # Make sure can compile
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
        # And try fitting... may be slow, though
        # Note that if flow is conditional, input_data will not be ignored
        history = m.fit(self.input_data, self.target_sample, epochs=1, verbose=0)
        assert history is not None
        # And test evaulation
        eval_loss = m.evaluate(self.input_data, self.target_sample, verbose=0)
        assert eval_loss is not None
        # And prediction
        pred = m.predict(self.input_data, verbose=0)
        assert pred is not None
        assert pred.shape == (self.input_data.shape[0], self.target_sample.shape[-1])

    @pytest.mark.parametrize("f", [
        flows.RQSSplineRealNVP(),
        flows.RQSSplineRealNVP(batch_norm=True),
        flows.RQSSplineMAF(),
        flows.RQSSplineMAF(batch_norm=True),
        flows.RQSSplineMAF(rqs_params={
            'conditional': True,
            'conditional_event_shape': input_data.shape[-1]
        }),
    ])
    def test_with_input(self, f):
        dep_dist = dists.AutoregressiveBlockwise(3, [tfp.distributions.Normal] * 2 + [tfp.distributions.VonMises])
        if f.conditional:
            _ = f(self.target_sample, conditional_input=self.input_data)
        else:
            _ = f(self.target_sample)
        m = models.FlowModel(f, dep_dist)
        assert m.mapping is not None
        _ = m(self.input_data)  # Make sure can pass through model
        # Make sure can compile
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
        # And try fitting... may be slow, though
        history = m.fit(self.input_data, self.target_sample, epochs=1, verbose=0)
        assert history is not None
        # And test evaulation
        eval_loss = m.evaluate(self.input_data, self.target_sample, verbose=0)
        assert eval_loss is not None
        # And prediction
        pred = m.predict(self.input_data, verbose=0)
        assert pred is not None
        assert pred.shape == (self.input_data.shape[0], self.target_sample.shape[-1])


class TestMappingToDistribution:

    input_data = tf.random.normal([10, 4])

    def test_creation_vonmises(self):
        vm = dists.IndependentVonMises(2)
        mtd_vm = models.MappingToDistribution(vm)
        out_vm = mtd_vm(self.input_data)
        assert not mtd_vm.conditional
        assert isinstance(mtd_vm.mapping, mappings.FCDeepNN)
        assert mtd_vm.mapping.target_shape == (6, )
        assert isinstance(out_vm, tfp.distributions.Distribution)

    def test_creation_normal(self):
        norm = tfp.layers.IndependentNormal(2)
        mtd_norm = models.MappingToDistribution(norm)
        out_norm = mtd_norm(self.input_data)
        assert mtd_norm.mapping.target_shape == (4, )
        assert isinstance(out_norm, tfp.distributions.Distribution)

    def test_creation_indblockwise(self):
        ib = dists.IndependentBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3)
        mtd_ib = models.MappingToDistribution(ib)
        out_ib = mtd_ib(self.input_data)
        assert mtd_ib.mapping.target_shape == (15, )
        assert isinstance(out_ib, tfp.distributions.Distribution)

    def test_creation_autoblockwise(self):
        ab = dists.AutoregressiveBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3)
        mtd_ab = models.MappingToDistribution(ab)
        out_ab = mtd_ab(self.input_data)
        assert mtd_ab.mapping.target_shape == mtd_ab.distribution.params_size()
        assert isinstance(out_ab, tfp.distributions.Distribution)

    def test_creation_auto_conditional(self):
        ab = dists.AutoregressiveBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3,
                                           conditional=True,
                                           conditional_event_shape=self.input_data.shape[-1])
        mtd_ab = models.MappingToDistribution(ab)
        assert mtd_ab.conditional
        out_ab = mtd_ab(self.input_data)
        assert mtd_ab.mapping.target_shape == mtd_ab.distribution.params_size()
        assert isinstance(out_ab, tfp.distributions.Distribution)

    def test_creation_flowed(self):
        fd = dists.FlowedDistribution(
            flows.RQSSplineRealNVP(),
            dists.IndependentBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3))
        _ = fd.flow(tf.ones([1, 6]))  # Build flow
        mtd_fd = models.MappingToDistribution(fd)
        out_fd = mtd_fd(self.input_data)
        assert mtd_fd.mapping.target_shape[0] == mtd_fd.distribution.params_size()
        assert isinstance(out_fd, tfp.distributions.Distribution)

    def test_creation_flowed_conditional(self):
        fd = dists.FlowedDistribution(
            flows.RQSSplineMAF(rqs_params={
                'conditional': True,
                'conditional_event_shape': self.input_data.shape[-1]
            }), dists.IndependentBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3))
        _ = fd.flow(tf.ones([1, self.input_data.shape[-1]]),
                    conditional_input=tf.ones([1, self.input_data.shape[-1]]))  # Build flow
        mtd_fd = models.MappingToDistribution(fd)
        assert mtd_fd.conditional
        out_fd = mtd_fd(self.input_data)
        assert mtd_fd.mapping.target_shape[0] == mtd_fd.distribution.params_size()
        assert isinstance(out_fd, tfp.distributions.Distribution)


class TestVAE:

    x = tf.random.normal((10, 6))
    zdim = 2

    def test_basic(self):
        encoder_dist = tfp.layers.IndependentNormal(self.zdim)
        encoder = models.MappingToDistribution(encoder_dist, name='encoder')
        decoder_dist = tfp.layers.IndependentNormal(self.x.shape[-1])
        decoder = models.MappingToDistribution(decoder_dist, name='decoder')
        # For simple prior with no flow, just use a distribution lambda
        prior = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros((self.zdim, )), scale=tf.ones((self.zdim, ))),
            reinterpreted_batch_ndims=1))
        vae = models.VAE(encoder, decoder, prior)
        out = vae(self.x)
        assert isinstance(vae.regularizer, losses.KLDivergenceEstimate)
        assert isinstance(out, tfp.distributions.Distribution)
        _ = out.sample()  # Check sampling
        _ = out.log_prob(self.x)  # And make sure log-probability works
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
        history = vae.fit(self.x, self.x, epochs=1, verbose=0)
        assert history is not None
        eval_loss = vae.evaluate(self.x, verbose=0)
        assert eval_loss is not None

    # Just testing a number of decoders
    # Could test all combos of encoders, priors, decoders, but probably too much
    # Decoders are what can change the most
    @pytest.mark.parametrize('decoder_dist', [
        dists.IndependentBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3),
        dists.AutoregressiveBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3),
        dists.AutoregressiveBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3,
                                      conditional=True,
                                      conditional_event_shape=2),
        dists.FlowedDistribution(
            flows.RQSSplineMAF(),
            dists.IndependentBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3)),
    ])
    def test_prior_flow_vary_decoders(self, decoder_dist):
        encoder_dist = dists.IndependentVonMises(self.zdim)
        encoder_map = mappings.FCDeepNN(encoder_dist.params_size(self.zdim), periodic_dofs=([False] * 3 + [True] * 3))
        encoder = models.MappingToDistribution(encoder_dist, mapping=encoder_map, name='encoder')
        decoder = models.MappingToDistribution(decoder_dist, name='decoder')
        # Build decoder flow if has one
        if hasattr(decoder_dist, 'flow'):
            if decoder_dist.conditional:
                _ = decoder_dist.flow(tf.ones((1, self.x.shape[-1])), tf.ones((1, self.zdim)))
            else:
                _ = decoder_dist.flow(tf.ones((1, self.x.shape[-1])))
        prior = dists.FlowedDistribution(
            flows.RQSSplineRealNVP(rqs_params={'bin_range': [-np.pi, np.pi]}),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(tfp.distributions.VonMises(
                loc=tf.zeros((self.zdim, )), concentration=tf.ones((self.zdim, ))),
                                                                                  reinterpreted_batch_ndims=1)),
            name='prior')
        _ = prior.flow(tf.ones((1, self.zdim)))  # Pass through flow to correctly build
        vae = models.VAE(encoder, decoder, prior)
        out = vae(self.x)
        assert isinstance(vae.regularizer, losses.KLDivergenceEstimate)
        assert isinstance(out, tfp.distributions.Distribution)
        _ = out.sample()  # Check sampling
        _ = out.log_prob(self.x)  # And make sure log-probability works
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
        history = vae.fit(self.x, self.x, epochs=1, verbose=0)
        assert history is not None
        eval_loss = vae.evaluate(self.x, verbose=0)
        assert eval_loss is not None

    def test_batch_norm(self):
        encoder_dist = tfp.layers.IndependentNormal(self.zdim)
        encoder_map = mappings.FCDeepNN(encoder_dist.params_size(self.zdim),
                                        batch_norm=True,
                                        periodic_dofs=([False] * 3 + [True] * 3))
        encoder = models.MappingToDistribution(encoder_dist, mapping=encoder_map, name='encoder')
        decoder_dist = dists.FlowedDistribution(
            flows.RQSSplineMAF(rqs_params={
                'conditional': True,
                'conditional_event_shape': self.zdim
            }, batch_norm=True),
            dists.IndependentBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3))
        decoder_dist.flow(tf.ones((1, self.x.shape[-1])), conditional_input=tf.random.uniform((1, self.zdim)))
        decoder_map = mappings.FCDeepNN(decoder_dist.params_size(), batch_norm=True)
        decoder = models.MappingToDistribution(decoder_dist, mapping=decoder_map, name='decoder')
        prior = dists.FlowedDistribution(
            flows.RQSSplineRealNVP(batch_norm=True),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(tfp.distributions.Normal(
                loc=tf.zeros((self.zdim, )), scale=tf.ones((self.zdim, ))),
                                                                                  reinterpreted_batch_ndims=1)),
            name='prior')
        prior.flow(tf.ones((1, self.zdim)))  # Build prior flow
        vae = models.VAE(encoder, decoder, prior)
        out = vae(self.x)
        assert isinstance(vae.regularizer, losses.KLDivergenceEstimate)
        assert isinstance(out, tfp.distributions.Distribution)
        _ = out.sample()  # Check sampling
        _ = out.log_prob(self.x)  # And make sure log-probability works
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
        history = vae.fit(self.x, self.x, epochs=1, verbose=0)
        assert history is not None
        eval_loss = vae.evaluate(self.x, verbose=0)
        assert eval_loss is not None


def test_backmappingonly():

    ref_coords = tf.random.uniform((200, 1, 3), -5.0, 5.0, dtype=np.float32)
    n_particles = np.random.randint(0, 50, size=200)
    fg_coords = np.vstack([np.random.uniform(-5.0, 5.0, (n, 3)) for n in n_particles])
    fg_coords = tf.RaggedTensor.from_row_splits(fg_coords, np.hstack([0, np.cumsum(n_particles)]))
    cg_coords = np.vstack([np.random.uniform(-5.0, 5.0, (n, 3)) for n in n_particles // 3])
    cg_coords = tf.RaggedTensor.from_row_splits(cg_coords, np.hstack([0, np.cumsum(n_particles // 3)]))
    fg_info = np.vstack([np.tile(np.array([1, 0], dtype=np.float32), (n, 1)) for n in n_particles])
    fg_info = tf.RaggedTensor.from_row_splits(fg_info, np.hstack([0, np.cumsum(n_particles)]))
    cg_info = np.vstack([np.tile(np.array([0, 1], dtype=np.float32), (n, 1)) for n in n_particles // 3])
    cg_info = tf.RaggedTensor.from_row_splits(cg_info, np.hstack([0, np.cumsum(n_particles // 3)]))
    all_coords = tf.concat([fg_coords, cg_coords], axis=1)
    all_info = tf.concat([fg_info, cg_info], axis=1)

    # Create masking and embedding for model
    mask_dist = mappings.DistanceSelection(3.0, max_included=10, box_lengths=[10.0, 10.0, 10.0])
    pe = mappings.ParticleEmbedding(20)
    mask_and_embed = mappings.LocalParticleDescriptors(mask_dist, pe)

    # Create decoder distribution for model
    decoder_dist = dists.AutoregressiveBlockwise(6, [tfp.distributions.Normal] * 3 + [tfp.distributions.VonMises] * 3)
    decoder = models.MappingToDistribution(decoder_dist, name='decoder')

    # Full model
    backmap = models.BackmappingOnly(mask_and_embed, decoder)

    out = backmap([ref_coords, all_coords, all_info])
    assert isinstance(out, tfp.distributions.Distribution)
    sample = out.sample()  # Check sampling
    _ = out.log_prob(sample)  # And check if can compute log-probability

    # Set target data and train on it
    target_data = tf.random.uniform((200, sample.shape[-1]), -5.0, 5.0, dtype=tf.float32)
    backmap.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
    history = backmap.fit([ref_coords, all_coords, all_info], target_data, batch_size=20, epochs=1, verbose=0)
    assert history is not None
    # And test evaulation
    eval_loss = backmap.evaluate([ref_coords, all_coords, all_info], target_data, batch_size=20, verbose=0)
    assert eval_loss is not None
    # And prediction
    pred = backmap.predict([ref_coords, all_coords, all_info], verbose=0)
    assert pred is not None
    assert pred.shape == (ref_coords.shape[0], 6)
