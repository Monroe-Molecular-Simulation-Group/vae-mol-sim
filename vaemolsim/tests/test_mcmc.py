"""
Tests for MCMC module.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from vaemolsim import losses, models, mcmc


class TestMCMC:

    x = tf.random.normal((10, 6))
    zdim = 2

    encoder_dist = tfp.layers.IndependentNormal(zdim)
    encoder = models.MappingToDistribution(encoder_dist, name='encoder')
    decoder_dist = tfp.layers.IndependentNormal(x.shape[-1])
    decoder = models.MappingToDistribution(decoder_dist, name='decoder')
    prior = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.zeros((tf.shape(t)[0], 2)), scale=tf.ones((tf.shape(t)[0], 2)))))
    vae = models.VAE(encoder, decoder, prior)
    _ = vae(x)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.LogProbLoss())
    vae.fit(x, x, epochs=1, verbose=0)

    @staticmethod
    def energy_func(configs):
        # Simple test energy function - just Gaussian with different means in each dimension
        means = np.linspace(-2, 2, 6)[np.newaxis, :]
        return np.sum((configs - means)**2, axis=-1)

    def test_creation(self):
        mc_chain = mcmc.MCMC(self.vae, self.energy_func)
        assert mc_chain._num_trials == 0
        assert mc_chain._num_acc == 0

    def test_single_step(self):
        mc_chain = mcmc.MCMC(self.vae, self.energy_func)
        # Single step without energies provided
        out_x, out_energies = mc_chain.single_step(self.x)
        assert out_x.shape == self.x.numpy().shape
        assert out_energies.shape[0] == out_x.shape[0]
        assert mc_chain._num_trials == self.x.shape[0]
        # Next single step with energies provided
        out_x, out_energies = mc_chain.single_step(out_x, energies=out_energies)
        assert mc_chain._num_trials == 2 * self.x.shape[0]

    def test_run(self):
        mc_chain = mcmc.MCMC(self.vae, self.energy_func)
        out_x, out_energies = mc_chain.run(self.x, n_steps=10)
        assert out_x.shape == self.x.numpy().shape
        assert out_energies.shape[0] == out_x.shape[0]
        assert mc_chain._num_trials == 10 * self.x.shape[0]
        # Test computing acceptance rate
        acc_rate = mc_chain.acceptance_rate
        assert acc_rate == (mc_chain._num_acc / mc_chain._num_trials)
        assert acc_rate <= 1.0
