"""
Defines VAE (or partial VAE, like just the encoder or decoder) models.

For convenience, a layer class is also defined that combines a mapping
with a distribution-creation layer. This is the common structure of
both encoders and decoders, which then streamlines the creation of
VAE models.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from . import dists, mappings, losses


class FlowModel(tf.keras.Model):
    """
  Model wrapper around FlowedDistribution layer so can quickly train in isolation.

  The output of a call to this model is a distribution transformed by the flow.
  As such, the appropriate loss is the losses.LogProbLoss(), or something else that
  will use a distribution that is provided.

  The primary purpose is training a flow to only transform from one distribution to another
  and that is the only objective. Specifically, if you have samples from a distribution you
  want to learn how to generate so that the loss is the log-probability of those samples
  under the flowed distribution (latent distribution passed through the flow).

  Attributes
  ----------
  flow : tfp.bijector object or layer applying a bijector
      Flow that is applied to transform the latent distribution.
  latent_dist : tfp.layers.Distribution object
      Distribution layer that produces distribution that flow acts on.
      The inputs to a call will be provided to latent_dist to produce the latent
      distribution. If the inputs should be ignored (i.e., the latent distribution
      is static), this should be a DistrbutionLambda layer with a make_distribution_fn
      that ignores inputs EXCEPT for the batch shape (if you want the produced
      distribution to have the same batch shape as the target).
      For example:
      static_dist = tfp.layers.DistributionLambda(make_distribution_fn=lambda t: tfp.distributions.Blockwise(
        [tfp.distributions.Normal(loc=tf.zeros((tf.shape(t)[0],)), scale=tf.ones((tf.shape(t)[0],)))]*2
        + [tfp.distributions.VonMises(loc=tf.zeros((tf.shape(t)[0],)), concentration=tf.ones((tf.shape(t)[0],)))],
                        ))
      Note that getting the batch shape right is not strictly necessary. A distribution
      with no batch shape will produce a log-probability for each sample of a batch the
      same as a distribution with a batch shape. However, prediction will work better
      if the batch shape is taken from the inputs. Otherwise, a batch only produces a
      single sample.
  mapping : layer, default None
      A layer mapping inputs to the input for a distribution creation layer. If not specified
      (default of None) will infer this from the params_size attribute of the distribution
      creation layer.
  """

    def __init__(self, flow, latent_dist, mapping=None, name='flow_model', **kwargs):
        """
    Creates a FlowModel model instance

    Parameters
    ----------
    flow : tfp.bijector object or layer applying bijector
        Flow that will be applied in order to transform latent distribution.
    latent_dist : tfp.layers.Distribution object
        Layer that produces distribution that flow acts upon in the forward
        direction to produce samples.
    mapping : layer, default None
        A layer mapping inputs to the input for a distribution creation layer. If not specified
        (default of None) will infer this from the params_size attribute of the distribution
        creation layer.
    """
        super(FlowModel, self).__init__(name=name, **kwargs)
        self.flowed_dist = dists.FlowedDistribution(flow, latent_dist)
        if mapping is None:
            if isinstance(latent_dist, tfp.layers.DistributionLambda):
                self.mapping = mapping
            else:
                self.mapping = mappings.FCDeepNN(self.flowed_dist.params_size())
        else:
            if isinstance(latent_dist, tfp.layers.DistributionLambda):
                print("Warning: for static distribution (tfp.layers.DistributionLambda as "
                      "latent distribution, cannot have mapping, so setting to None.")
                self.mapping = None

    def call(self, inputs, training=False):
        """
    Applies a flow to the latent distribution.

    Parameters
    ----------
    inputs : tf.Tensor
        Samples from the distribution we want to learn (forward application of the flow
        produces this distribution). If latent_dist is a distribution-producing layer,
        the inputs will be used by it to produce a distribution (likely including a
        MappingToDistribution layer).
    conditional_input : tf.Tensor
        Optional conditional input to provide to the flow.
    training : bool, default False
        Whether or not we are training. May be important if have block normalizations in flow layer.

    Returns
    -------
    dist : tfp.distributions.TransformedDistribution instance
    """
        # Apply mapping if present
        if self.mapping is not None:
            mapped = self.mapping(inputs, training=training)
        else:
            mapped = inputs

        # And pass through flow
        if self.flowed_dist.conditional:
            return self.flowed_dist(mapped, conditional_input=inputs, training=training)
        else:
            return self.flowed_dist(mapped, training=training)

    def predict_step(self, inputs):
        """
    A custom predict step makes the model more useful.

    fit and evaluate will compute losses (and metrics) with training set to True and False, respectively.
    The returned loss will typically be the average negative log-probability over the batch.
    predict will actually draw a sample from the learned flowed distribution.
    You can always access whatever you need based on model.flowed_dist.flow and model.flowed_dist.latent_dist.
    Or by calling the model on inputs, but this is a short-cut.

    Parameters
    ----------
    inputs : tf.Tensor
        Inputs to the model.

    Returns
    -------
    z : tf.Tensor
        A transformed sample under flow.
    """
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        out_dist = self(x, training=False)
        return out_dist.sample()

    def get_config(self):
        config = super(FlowModel, self).get_config()
        config.update({
            "flow": self.flowed_dist.flow,
            "latent_dist": self.flowed_dist.latent_dist,
            "mapping": self.mapping,
        })
        return config


# If reconfigure this right, can allow mappings that include a masking and embedding...
# Maybe that's too much, but would be possible
class MappingToDistribution(tf.keras.Model):
    """
  A combination of a mapping layer and a distribution creation layer, bundled into a model.

  This is likely to represent an encoder or a decoder, where a mapping or projection
  is applied with neural networks before defining a probability distribution. This
  just packages both together to help more modularly define a VAE model. It also
  defines default behavior here.

  Attributes
  ----------
  distribution : layer
      A layer to create a tfp.distribution object.
  mapping : layer
      A layer mapping inputs to this layer to the inputs of the distribution-creation layer.
  conditional : bool
      Whether or not conditional inputs will be passed to the distribution-creation layer.
  """

    def __init__(self, distribution, mapping=None, name='map_to_dist', **kwargs):
        """
    Creates a MappingToDistribution layer instance

    Parameters
    ----------
    distribution : layer
        A layer that creates a tfp.distributions object given inputs.
    mapping : layer, default None
        A layer mapping inputs to the input for a distribution creation layer. If not specified
        (default of None) will infer this from the params_size attribute of the distribution
        creation layer.
    """
        super(MappingToDistribution, self).__init__(name=name, **kwargs)

        self.distribution = distribution

        # Check if distribution layer takes conditional inputs
        # Seems a little clunky to have to have a conditional attribute and read it here...
        # No good alternative at present
        try:
            self.conditional = self.distribution.conditional
        except AttributeError:
            self.conditional = False

        # If mapping is not defined, create it so that it works with the provided distribution
        if mapping is None:
            if issubclass(type(self.distribution), tfp.layers.DistributionLambda):
                self.mapping = mappings.FCDeepNN(self.distribution.params_size(self.distribution._event_shape))
            else:
                self.mapping = mappings.FCDeepNN(self.distribution.params_size())
        else:
            self.mapping = mapping

    def call(self, inputs, training=False):
        """
      Applies mapping to distribution parameters and uses these to create a tfp.distributions object.

    Parameters
    ----------
    inputs : tf.Tensor
        Inputs to this layer.
    training : bool, default False
        Whether training or predicting. Applicable if have batch normalizations or other behaviors as
        part of either the mapping or distribution creation layers.

    Returns
    -------
    tfp.distributions object
    """
        # Note that mapping takes same inputs as conditional_inputs...
        # Hard to handle situations that need more flexibility, though, especially if inputs is a list
        # But think about how to make more general so can accomodate things like the distance-based masking
        mapped = self.mapping(inputs, training=training)
        if self.conditional:
            return self.distribution(mapped, training=training, conditional_input=inputs)
        else:
            return self.distribution(mapped, training=training)

    def get_config(self):
        config = super(MappingToDistribution, self).get_config()
        config.update({
            "distribution": self.distribution,
            "mapping": self.mapping,
        })
        return config


# Consider adding custom predict step to generate x samples from the model
# (e.g., draw from prior, produce decoder distribution, and draw from that)
class VAE(tf.keras.Model):
    """
  A standard variational autoencoder model.

  Input is passed through an encoder to produce an encoding probability distribution. A sample is
  drawn from that distribution and a regularization loss is computed based on the encoder, prior,
  and encoder sample. The encoder sample is then passed through a decoder to produce a decoding
  probability distribution. This is passed, along with inputs, to a reconstruction loss function
  that is provided at compile time.

  Attributes
  ----------
  encoder : layer
      Maps inputs to an encoder distribution.
  decoder : layer
      Maps encoded values to a decoder distribution.
  prior : layer or tf.Module
      Defines the prior distribution.
  regularizer : callable
      Defines the regularization loss to add to the model.
  """

    def __init__(self, encoder, decoder, prior, regularizer=losses.KLDivergenceEstimate(), name='vae', **kwargs):
        """
    Creates a VAE model instance

    Parameters
    ----------
    encoder : keras layer
        A layer that maps from inputs to an encoder distribution (tfp.distributions object).
    decoder : keras layer
        A layer that maps from latent samples to a decoder distribution (tfp.distributions object).
    prior : keras layer or tf.Module
        A layer or module that, when called, produces the prior distribution (requiring a
        call ensures that batch normalization bijectors have their training attribute set correctly).
    regularizer : regularizer, default vaemolsim.losses.KLDivergenceEstimate
        A regularization loss, like a KL divergence between the encoder and prior, which is the default.
        Whatever callable is used, it should follow the definitions for the call signature described
        when subclassing vaemolsim.losses.InfoRegularizer (subclassing this is recommended).
    """
        super(VAE, self).__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.regularizer = regularizer

    def call(self, inputs, training=False):
        """
      Encodes, regularizes to a prior, and decodes, producing a decoder probability distribution.

      Parameters
      ----------
      inputs : tf.Tensor
          Inputs to the VAE, which should also be samples of the distribution we want to model.
      training : bool, default False
          Whether or not performing training or prediction.

      Returns
      -------
      decode_dist : tfp.distributions object
          A tfp.distributions object that represents the model for the decoder probability distribution
          for the provided sample. This can be used to easily assess the loss by calling its log_prob
          method on the input samples.
    """

        encode_dist = self.encoder(inputs, training=training)
        # Consider adding parameter to control sampling more than just once per input in the batch?
        encode_sample = encode_dist.sample()

        # Build prior distribution - should only use encode_sample to build correct shape, if at all
        prior_dist = self.prior(encode_sample, training=training)

        reg_loss = self.regularizer(encode_dist, prior_dist, encode_sample)
        self.add_loss(reg_loss)
        self.add_metric(reg_loss / self.regularizer.weight, name='kl_div', aggregation='mean')
        self.add_metric(reg_loss, name='regularizer_loss', aggregation='mean')

        decode_dist = self.decoder(encode_sample, training=training)

        return decode_dist

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "prior": self.prior,
            "regularizer": self.regularizer,
        })
        return config


class VAEDualELBO(tf.keras.Model):
    """
  A variational autoencoder model but intended to be trained with both forward and reverse ELBO losses.

  This means the model traverses two directions (x to z to x, the standard, and (z to x to z the reverse).
  The model will thus have two outputs and require two reconstruction losses, one for the standard ELBO
  and one involving the potential energy for the reverse. We will also have two regularizers, with the
  typical being the KL divergence for the forward and the KL divergence for the reverse. It is intended
  to be used with the loss functions (for reconstruction losses) LogProbLoss and PotentialEnergyLogProbLoss
  for the forward and reverse ELBO estimates, respectively.

  Attributes
  ----------
  encoder : layer
      Maps inputs to an encoder distribution.
  decoder : layer
      Maps encoded values to a decoder distribution.
  prior : layer or tf.Module
      Defines the prior distribution.
  regularizer_forward : callable
      Defines the regularization loss to add to the model on the forward pass.
  regularizer_reverse : callable
      Defines the regularization loss to add to the model on the reverse pass.
  """

    def __init__(self,
                 encoder,
                 decoder,
                 prior,
                 regularizer_forward=losses.KLDivergenceEstimate(),
                 regularizer_reverse=losses.ReverseKLDivergenceEstimate(),
                 name='vae_dual',
                 **kwargs):
        """
    Creates a VAEDualELBO model instance

    Parameters
    ----------
    encoder : layer
        A layer that maps from inputs to an encoder distribution.
    decoder : layer
        A layer that maps from latent samples to a decoder distribution.
    prior : layer
        A layer or module that, when called, produces the prior distribution.
        Requiring a call ensures that batch normalization bijectors have their training
        attribute set correctly.
    regularizer_forward : regularizer, default losses.KLDivergenceEstimate
        A regularization loss, like a KL divergence between the encoder and prior, which is the default.
        Whatever callable is used, it should follow the definitions for the call signature described when
        subclassing losses.InfoRegularizer (subclassing this is recommended). Note that this regulaerizer
        only applies to forward direction of ELBO.
    regularizer_reverse : layer, default losses.ReverseKLDivergenceEstimate
        Applies to reverse direction of ELBO.
    """
        super(VAE, self).__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.regularizer_forward = regularizer_forward
        self.regularizer_reverse = regularizer_reverse

    def call(self, inputs, training=False):
        prior_dist = self.prior(None, training=training)

        # Forward pass (x to z to x)
        encode_dist_forward = self.encoder(inputs, training=training)
        # Consider adding parameter to control sampling more than just once per input in the batch?
        encode_sample = encode_dist_forward.sample()
        decode_dist_forward = self.decoder(encode_sample, training=training)
        reg_loss_forward = self.regularizer_forward(encode_dist_forward, prior_dist, encode_sample)
        self.add_loss(reg_loss_forward)
        self.add_metric(reg_loss_forward / self.regularizer_forward.weight, name='kl_div_forward', aggregation='mean')
        self.add_metric(reg_loss_forward, name='regularizer_loss_forward', aggregation='mean')

        # Backward pass (z to x to z)
        prior_sample = prior_dist.sample()
        decode_dist_reverse = self.decoder(prior_sample, training=training)
        decode_sample = decode_dist_reverse.sample()
        encode_dist_reverse = self.encoder(decode_sample, training=training)
        reg_loss_reverse = self.regularizer_reverse(encode_dist_reverse, prior_dist, prior_sample)
        self.add_loss(reg_loss_reverse)
        self.add_metric(reg_loss_reverse / self.regularizer_reverse.weight, name='kl_div_reverse', aggregation='mean')
        self.add_metric(reg_loss_reverse, name='regularizer_loss_reverse', aggregation='mean')

        return [decode_dist_forward, decode_dist_reverse]

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "encoder": self.encoder,
            "decoder": self.decoder,
            "prior": self.prior,
            "regularizer": self.regularizer,
        })
        return config


# Need a model for decoding with spatial convolutions over residues or molecules
# Workflow should go something like:
#  1) Encoder maps deterministically to CG configuration
#  2) Randomly permute CG molecules/residues, saving permutation order so can go back at end
#  3) For each CG site, apply molecule/residue-specific backmapping
#      a) Identify nearby CG sites and pass through SchNet type architecture to produce part of decoder input
#      b) Identify nearby atoms (already decoded) and pass through SchNet type architecture to produce part of
#         decoder input
#      c) Concatenate decoder input information, which should have fixed size somehow
#      d) Perform decoding for JUST molecule/residue atoms given nearby CG sites and atoms (decoder input)
#      e) Accumulate probability distribution of each newly decoded molecule/residue as go (maybe as complex
#         JointDistribution?)
#  4) Somehow compute loss from output of the decoder...
# There are two options for how to think about what the decoder outputs.
# We can stick with what we've been doing and return a complex probability distribution object.
# Clearly, this would be complex, and really wrap the entirety of step 3, having CG configurations and samples
# from previously
# created distributions passed through a SchNet architecture to predict the parameters for the next distribution.
# In this way, the distribution would be autoregressive and would be convolutional due to the distance-dependence
# of SchNet.
# The other way is to handle all of this without creating a complicated decoder object and just returning
# log-probabilities.
# The downside there is that it is set up for training and you will need a different set of code for sampling.
# JointDistributionSequential seems to make the most sense.
# Each distribution is created based on the previous distributions outputs and the CG configuration.
# Instead of lambda functions, the callable to create the next distribution should choose the right
# molecule/residue model and feed ALL the previous inputs and CG configuration into it to create the next
# distribution.
# The tricky part is getting ALL the previous inputs fed in, which is where an Autoregressive has an advantage.
# However, we need JointDistribution because it can have different numbers of dofs sampled with different
# distributions and even nest autoregressive distributions or flows inside of itself.
# For log-probabilities of inputs with a joint distribution, just need to create a special LogProbLoss-like
# class that also has a parameter to map inputs in (N_atoms, 3) format to the per-molecule/residue local
# coordinate system used for decoding.
# Alternatively, could apply this map before training and pass as the y argument during fit.


class BackmappingOnly(tf.keras.Model):
    """
  Model to only apply a backmapping, assuming encoding is taken care of by pre-processing.

  This means that the inputs will be CG (and potentially FG) coordinates, along with particle
  properties (and potentially simulation box information). The outputs are expected to be
  distributions over the local coordinates of the CG beads to decode. Note that inputs
  should be in a list of [CG_coords_to_decode, other_CG_and_FG_coords, other_particle_props].
  Outputs are JUST the local FG coordinates corresponding to the CG coordinates intended for
  decoding (like sidechain BAT coordinates), which will allow a LogProbLoss to be used easily.
  Lots of work is required here in preprocessing, but the overall model is simpler. The
  downside is that a loop over separate trained models will be needed to decode different
  molecules or residues if that is desired.

  Attributes
  ----------
  mask_and_embed : layer
      Layer that identifies and embeds the local point cloud environment around a reference point.
  decode_dist : layer
      Layer that produces a decoder distribution.
  """

    def __init__(self, mask_and_embed, decode_dist, name='backmapping', **kwargs):
        """
    Creates BackmappingOnly model instance

    Parameters
    ----------
    mask_and_embed : layer
        A layer that converts CG and FG coords, along with properties to local embeddings around
        each CG bead to decode. Should take three arguments as input, with the order being
        other_CG_and_FG_coords, CG_coords_to_decode, other_particle_props (like LocalParticleDescriptors layer).
    decode_dist : layer
        A layer to take local embeddings and convert them to tfp.distribution objects to output (like a
        MappingToDistribution layer).
    """
        super(BackmappingOnly, self).__init__(name=name, **kwargs)
        self.mask_and_embed = mask_and_embed
        self.decode_dist = decode_dist

    def call(self, inputs, training=False):
        """
    From low-dimensional (CG) coordinates, produces a decoding probability distribution for fine-grained coordinates.

    Parameters
    ----------
    inputs : list
        Should be list of three tf.Tensor objects. The first are the CG beads to be decoded, which are of
        shape (N_batch, 1, 3). Next are all other CG or fine-grained coordinates (except those we want to
        generate) of shape (N_batch, N_particles, 3), where this can be a ragged tensor with N_particles
        being different for each batch member. Finally, additional particle information, like particle type
        (or one-hot encoding of this), particle parameters, etc., is the last element, with its shape
        matching the first two dimensions of the second element (so it can also be a ragged tensor).
    training : bool, default False
        Whether or not we are training or predicting (generating). This is necessary because many
        layers may have specialized training-only operations, like batch normalization.

    Returns
    -------
    decode_dist : tfp.distributions object
        The decoding distribution represent the probability distribution of the fine-grained degrees
        of freedom we want to conditionally predict from CG (or other already decoded fine-grained)
        information.
    """
        cg_to_decode = inputs[0]
        other_coords = inputs[1]
        other_particle_props = inputs[2]

        local_descriptors = self.mask_and_embed(other_coords, cg_to_decode, other_particle_props)

        decode_dist = self.decode_dist(local_descriptors, training=training)

        return decode_dist

    def predict_step(self, inputs):
        """
    A custom predict step to make generation (sampling) of configurations easier.

    fit and evaluate will compute losses (and metrics) with training set to True and False, respectively.
    The returned loss will typically be the average negative log-probability over the batch.
    predict will actually draw a sample from the learned distribution given inputs.

    Parameters
    ----------
    inputs : list of tf.Tensor
        Inputs to the model (see call method).

    Returns
    -------
    z : tf.Tensor
        A sample from the predicted distribution.
    """
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        out_dist = self(x, training=False)
        return out_dist.sample()

    def get_config(self):
        config = super(BackmappingOnly, self).get_config()
        config.update({
            "mask_and_embed": self.mask_and_embed,
            "decode_dist": self.decode_dist,
        })
        return config
