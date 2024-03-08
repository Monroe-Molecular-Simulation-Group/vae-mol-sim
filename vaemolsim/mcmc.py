"""
Defines classes and methods for performing VAE-based MC.
"""

import numpy as np


class MCMC(object):
    """
    Markov Chain Monte Carlo simulation object using a VAE for generating moves.

    Attributes
    ----------
    vae : vaemolsim.models.VAE object
        A trained VAE model for generating new configurations.
    energy_func : callable
        Given a simulation, or batch of configurations, computes the energy
        of all of them (or -log(probability))
    """

    def __init__(self, vae, energy_func, random_seed=None):
        """
        Creates a MCMC model instance

        Parameters
        ----------
        vae : vaemolsim.models.VAE object
            A trained VAE model for generating new configurations.
        energy_func : callable
            Given a simulation, or batch of configurations, computes the energy
            of all of them (or -log(probability))
        random_seed : int (default None)
            Optional random seed for controlling the seed for the random number generator.
            If not specified, just creates a new random number generator.
        """
        self.vae = vae
        self.energy_func = energy_func

        self._num_trials = 0.0
        self._num_acc = 0.0

        self._rng = np.random.default_rng(seed=random_seed)

    @property
    def acceptance_rate(self):
        return self._num_acc / self._num_trials

    def reset(self, random_seed=None):
        """
        Resets MC statistics.

        Parameters
        ----------
        random_seed : int (default None)
            Optional random seed for controlling the seed for the random number generator.
            If not specified, just creates a new random number generator.
        """
        self._num_trials = 0.0
        self._num_acc = 0.0

        # Also reset the random number generator
        self._rng = np.random.default_rng(seed=random_seed)

    def single_step(self, configs, energies=None):
        """
        Takes single MC step given a set of starting configurations.

        Parameters
        ----------
        configs : np.ndarray or tf.Tensor
            Batch of configurations of shape (N_batch, N_DOFs) where the second
            dimension involves the number of degrees of freedom entering the
            energy function or the VAE
        energies : np.ndarray or tf.Tensor (default None)
            Optionally can provide the energies of the starting configurations.
            Should match first dimension of configs, N_batch.

        Returns
        -------
        new_configs : np.ndarray
            New configurations produce by proposing a new move and accepting or rejecting.
            If the move is rejected, this will be the same as the input configurations,
            though for a batch of configs, some may be rejected and accepted. Essentially,
            runs as many parallel MCMC chains as input configurations.
        new_energies : np.ndarray
            Energies of new configurations
        """
        # Make sure it's an array
        configs = np.array(configs)

        # If energies aren't provided, compute them
        if energies is None:
            energies = self.energy_func(configs)

        # Create new sample by moving to latent space, moving within, and decoding
        z1, log_z1_given_x1 = self.vae.encoder(configs).experimental_sample_and_log_prob()
        z2, log_z2 = self.vae.prior(z1).experimental_sample_and_log_prob()
        new_configs, log_x2_given_z2 = self.vae.decoder(z2).experimental_sample_and_log_prob()
        forward_log_p = (log_z1_given_x1 + log_z2 + log_x2_given_z2).numpy()

        # Evaluate reverse proposal probability
        log_z2_given_x2 = self.vae.encoder(new_configs).log_prob(z2)
        log_z1 = self.vae.prior(z2).log_prob(z1)
        log_x1_given_z1 = self.vae.decoder(z1).log_prob(configs)
        reverse_log_p = (log_z2_given_x2 + log_z1 + log_x1_given_z1).numpy()

        # Get new energies
        new_configs = new_configs.numpy()
        new_energies = self.energy_func(new_configs)

        # Compute acceptance log-probabilities
        log_acc = new_energies + reverse_log_p - energies - forward_log_p

        # Accept or reject
        log_rand = np.log(self._rng.random(size=log_acc.shape[0]))
        acc_bool = (log_acc >= log_rand)

        # Store statistics
        self._num_trials += len(acc_bool)
        self._num_acc += np.sum(acc_bool)

        # Replace new configurations with old where not accepted
        new_configs[~acc_bool, ...] = configs[~acc_bool, ...]
        new_energies[~acc_bool] = energies[~acc_bool]

        return new_configs, new_energies

    def run(self, configs, energies=None, n_steps=1):
        """
        Runs MCMC for specified number of steps.

        Parameters
        ----------
        configs : np.ndarray or tf.Tensor
            Batch of configurations of shape (N_batch, N_DOFs) where the second
            dimension involves the number of degrees of freedom entering the
            energy function or the VAE.
        energies : np.ndarray or tf.Tensor (default None)
            Optionally can provide the energies of the starting configurations.
            Should match first dimension of configs, N_batch.
        n_steps : int (default 1)
            Number of steps to run.

        Returns
        -------
        configs : np.ndarray
            Configurations at the end of the MC chain sampling.
        energies : np.ndarray
            Energies of final configurations.
        """
        for n in range(n_steps):
            configs, energies = self.single_step(configs, energies=energies)

        return configs, energies
