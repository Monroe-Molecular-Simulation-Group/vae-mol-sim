"""
Defines VAE (or partial VAE, like just the encoder or decoder) models.
For convenience, a layer class is also defined that combines a mapping
with a distribution-creation layer. This is the common structure of
both encoders and decoders, which then streamlines the creation of
VAE models.
"""

#Put overall models here
#As in, the full VAE

import tensorflow as tf
import tensorflow_probability as tfp

import mappings
import losses


#If reconfigure this right, can allow mappings that include a masking and embedding...
#Maybe that's too much, but would be possible
class MappingToDistribution(tf.keras.layers.Layer):
  """
  A combination of a mapping layer and a distribution creation layer. This is likely to
  represent an encoder or a decoder, where a mapping or projection is applied with neural
  networks before defining a probability distribution.

  This just packages both together to help more modularly define a VAE model. It also
  defines default behavior here.
  """

  def __init__(self,
               distribution,
               mapping=None,
               name='map_to_dist',
               **kwargs):
    """
    Inputs:
        distribution - a layer that creates a tfp.distributions object given inputs
        mapping - (None) a layer mapping inputs to the input for a distribution creation layer
    Outputs:
        MappingToDistribution layer
    """
    super(MappingToDistribution, self).__init__(name=name, **kwargs)
    
    self.distribution = distribution
    
    #Check if distribution takes conditional inputs
    #Seems a little clunky to have to have a conditional attribute and read it here...
    try:
      self.conditional = self.distribution.conditional
    except AttributeError:
      self.conditional = False
    
    #If mapping is not defined, create it so that it works with the provided distribution
    if mapping is None:
      if issubclass(type(self.distribution), tfp.layers.DistributionLambda):
        self.mapping = mappings.FCDeepNN(self.distribution.params_size(self.distribution._event_shape))
      else:
        self.mapping = mappings.FCDeepNN(self.distribution.params_size())
    else:
      self.mapping = mapping

  def call(self, inputs, training=False):
    #Note that mapping takes same inputs as conditional_inputs...
    #Hard to handle situations that need more flexibility, though, especially if inputs is a list
    #But think about how to make more general so can accomodate things like the distance-based masking
    mapped = self.mapping(inputs, training=training)
    if self.conditional:
      return self.distribution(mapped, training=training, conditional_input=inputs)
    else:
      return self.distribution(mapped, training=training)

  def get_config(self):
    config = super(MappingToDistribution, self).get_config()
    config.update({"distribution": self.distribution,
                   "mapping": self.mapping,
                  })
    return config


class VAE(tf.keras.Model):
  """
  A standard variational autoencoder model. Input is passed through an encoder to produce an encoding
  probability distribution. A sample is drawn from that distribution and a regularization loss is
  computed based on the encoder, prior, and encoder sample. The encoder sample is then passed through a
  decoder to produce a decoding probability distribution. This is passed, along with inputs, to a
  reconstruction loss function that is provided at compile time.
  """
  
  def __init__(self,
               encoder,
               decoder,
               prior,
               regularizer=losses.KLDivergenceEstimate(),
               name='vae',
               **kwargs):
    """
    Inputs:
        encoder - a layer that maps from inputs to an encoder distribution
        decoder - a layer that maps from latent samples to a decoder distribution
        prior - a layer or module that, when called, produces the prior distribution
                (requiring a call ensures that batch normalization bijectors have
                their training attribute set correctly)
        regularizer - (default losses.KLDivergenceEstimate) a regularization loss, like a
                      KL divergence between the encoder and prior, which is the default;
                      whatever callable is used, it should follow the definitions for the
                      call signature described when subclassing losses.InfoRegularizer
                      (subclassing this is recommended)
    Outputs:
        VAE model
    """
    super(VAE, self).__init__(name=name, **kwargs)

    self.encoder = encoder
    self.decoder = decoder
    self.prior = prior
    self.regularizer = regularizer

  def call(self, inputs, training=False):

    prior_dist = self.prior(None, training=training) #Layer, so needs inputs, but will ignore so pass None
    encode_dist = self.encoder(inputs, training=training)
    # Consider adding parameter to control sampling more than just once per input in the batch?
    encode_sample = encode_dist.sample()

    reg_loss = self.regularizer(encode_dist, prior_dist, encode_sample)
    self.add_loss(reg_loss)
    self.add_metric(reg_loss/self.regularizer.weight, name='kl_div', aggregation='mean') 
    self.add_metric(reg_loss, name='regularizer_loss', aggregation='mean') 

    decode_dist = self.decoder(encode_sample, training=training)
    
    return decode_dist

  def get_config(self):
    config = super(VAE, self).get_config()
    config.update({"encoder": self.encoder,
                   "decoder": self.decoder,
                   "prior": self.prior,
                   "regularizer": self.regularizer,
                  })
    return config


class VAEDualELBO(tf.keras.Model):
  """
  A variational autoencoder model but intended to be trained with both forward and reverse
  ELBO losses. This means the model traverses two directions (x to z to x, the standard, and
  (z to x to z the reverse). The model will thus have two outputs and require two reconstruction
  losses, one for the standard ELBO and one involving the potential energy for the reverse. We will
  also have two regularizers, with the typical being the KL divergence for the forward and the KL
  divergence for the reverse. It is intended to be used with the loss functions (for reconstruction
  losses) LogProbLoss and PotentialEnergyLogProbLoss for the forward and reverse ELBO estimates,
  respectively.
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
    Inputs:
        encoder - a layer that maps from inputs to an encoder distribution
        decoder - a layer that maps from latent samples to a decoder distribution
        prior - a layer or module that, when called, produces the prior distribution
                (requiring a call ensures that batch normalization bijectors have
                their training attribute set correctly)
        regularizer_forward - (default losses.KLDivergenceEstimate) a regularization loss, like a
                      KL divergence between the encoder and prior, which is the default;
                      whatever callable is used, it should follow the definitions for the
                      call signature described when subclassing losses.InfoRegularizer
                      (subclassing this is recommended); applies to forward direction of ELBO
        regularizer_reverse - (default losses.ReverseKLDivergenceEstimate) applies to reverse
                      direction of ELBO.
    Outputs:
        VAEDualELBO model
    """
    super(VAE, self).__init__(name=name, **kwargs)

    self.encoder = encoder
    self.decoder = decoder
    self.prior = prior
    self.regularizer_forward = regularizer_forward
    self.regularizer_reverse = regularizer_reverse

  def call(self, inputs, training=False):
    prior_dist = self.prior(None, training=training)

    #Forward pass (x to z to x)
    encode_dist_forward = self.encoder(inputs, training=training)
    # Consider adding parameter to control sampling more than just once per input in the batch?
    encode_sample = encode_dist_forward.sample()
    decode_dist_forward = self.decoder(encode_sample, training=training)
    reg_loss_forward = self.regularizer_forward(encode_dist_forward, prior_dist, encode_sample)
    self.add_loss(reg_loss_forward)
    self.add_metric(reg_loss_forward/self.regularizer_forward.weight, name='kl_div_forward', aggregation='mean') 
    self.add_metric(reg_loss_forward, name='regularizer_loss_forward', aggregation='mean') 

    #Backward pass (z to x to z)
    prior_sample = prior_dist.sample()
    decode_dist_reverse = self.decoder(prior_sample, training=training)
    decode_sample = decode_dist_reverse.sample()
    encode_dist_reverse = self.encoder(decode_sample, training=training)
    reg_loss_reverse = self.regularizer_reverse(encode_dist_reverse, prior_dist, prior_sample)
    self.add_loss(reg_loss_reverse)
    self.add_metric(reg_loss_reverse/self.regularizer_reverse.weight, name='kl_div_reverse', aggregation='mean') 
    self.add_metric(reg_loss_reverse, name='regularizer_loss_reverse', aggregation='mean') 
    
    return [decode_dist_forward, decode_dist_reverse]

  def get_config(self):
    config = super(VAE, self).get_config()
    config.update({"encoder": self.encoder,
                   "decoder": self.decoder,
                   "prior": self.prior,
                   "regularizer": self.regularizer,
                  })
    return config


#Need a model for decoding with spatial convolutions over residues or molecules
#Workflow should go something like:
#  1) Encoder maps deterministically to CG configuration
#  2) Randomly permute CG molecules/residues, saving permutation order so can go back at end
#  3) For each CG site, apply molecule/residue-specific backmapping
#      a) Identify nearby CG sites and pass through SchNet type architecture to produce part of decoder input
#      b) Identify nearby atoms (already decoded) and pass through SchNet type architecture to produce part of decoder input
#      c) Concatenate decoder input information, which should have fixed size somehow
#      d) Perform decoding for JUST molecule/residue atoms given nearby CG sites and atoms (decoder input)
#      e) Accumulate probability distribution of each newly decoded molecule/residue as go (maybe as complex JointDistribution?)
#  4) Somehow compute loss from output of the decoder...
#There are two options for how to think about what the decoder outputs.
#We can stick with what we've been doing and return a complex probability distribution object.
#Clearly, this would be complex, and really wrap the entirety of step 3, having CG configurations and samples from previously
#created distributions passed through a SchNet architecture to predict the parameters for the next distribution.
#In this way, the distribution would be autoregressive and would be convolutional due to the distance-dependence of SchNet.
#The other way is to handle all of this without creating a complicated decoder object and just returning log-probabilities.
#The downside there is that it is set up for training and you will need a different set of code for sampling.
#JointDistributionSequential seems to make the most sense.
#Each distribution is created based on the previous distributions outputs and the CG configuration.
#Instead of lambda functions, the callable to create the next distribution should choose the right molecule/residue model
#and feed ALL the previous inputs and CG configuration into it to create the next distribution.
#The tricky part is getting ALL the previous inputs fed in, which is where an Autoregressive has an advantage.
#However, we need JointDistribution because it can have different numbers of dofs sampled with different distributions and
#even nest autoregressive distributions or flows inside of itself.
#For log-probabilities of inputs with a joint distribution, just need to create a special LogProbLoss-like class that also
#has a parameter to map inputs in (N_atoms, 3) format to the per-molecule/residue local coordinate system used for decoding.
#Alternatively, could apply this map before training and pass as the y argument during fit.


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
  """

  def __init__(self,
               mask_and_embed,
               decode_dist,
               name='backmapping',
               **kwargs):
    """
    Inputs:
        mask_and_embed - a layer that converts CG and FG coords, along with properties to local embeddings around
                         each CG bead to decode; should take three arguments as input, with the order being
                         other_CG_and_FG_coords, CG_coords_to_decode, other_particle_props (like LocalParticleDescriptors layer)
        decode_dist - a layer to take local embeddings and convert them to tfp.distribution objects to output
                      (like a MappingToDistribution layer)
    Outputs:
        BackmappingOnly object
    """
    super(BackmappingOnly, self).__init__(name=name, **kwargs)
    self.mask_and_embed = mask_and_embed
    self.decode_dist = decode_dist

  def call(self, inputs, training=False):
    cg_to_decode = inputs[0]
    other_coords = inputs[1]
    other_particle_props = inputs[2]

    local_descriptors = self.mask_and_embed(other_coords, cg_to_decode, other_particle_props)

    decode_dist = self.decode_dist(local_descriptors, training=training)

    return decode_dist

  def get_config(self):
    config = super(LocalParticleDescriptors, self).get_config()
    config.update({"mask_and_embed": self.mask_and_embed,
                   "decode_dist": self.decode_dist,
                  })
    return config

