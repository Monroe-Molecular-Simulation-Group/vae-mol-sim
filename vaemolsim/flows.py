"""
Defines normalizing flows based on tensorflow-probability bijectors.

All classes inherit from tf.keras.layers.Layer, except for a single
tf.keras.Model class defined here for convenience of training only
normalizing flows.
"""

import tensorflow as tf
import tensorflow_probability as tfp


class SplineBijector(tf.keras.layers.Layer):
    """
  Layer to implement a spline bijector function.

  Follows tfp example for using rational quadratic splines (as described in Durkan et al.
  2019) to replace affine transformations (in, say, RealNVP). This should allow more flexible
  transformations with similar cost and should work much better with 1D flows. Intended for
  operation with tfp.bijectors.RealNVP, which impacts the format of the "call" method.

  Attributes
  ----------
  data_dim : int
      Dimension of the input data, which will also be the dimension of the output.
  bin_min : int
      Minimum value for spline to act on data.
  bin_max : int
      Maximum value for spline to act on data.
  num_bins : int
      Number of spline bins (intervals).
  hidden_dim : int
      Dimensionality of any hidden neural network layers.
  """

    def _bin_positions(self, x):
        """
    Defines positions of bins between spline knots as an activation to a dense network's output.
    """
        x = tf.reshape(x, [tf.shape(x)[0], -1, self.num_bins])
        out = tf.math.softmax(x, axis=-1)
        out = out * (self.bin_max - self.bin_min - self.num_bins * 1e-2) + 1e-2
        return out

    def _slopes(self, x):
        """
    Defines slopes of splines over each bin.
    Acts as a activation over a dense network's output.
    """
        x = tf.reshape(x, [tf.shape(x)[0], -1, self.num_bins - 1])
        return tf.math.softplus(x) + 1e-2

    def __init__(self,
                 data_dim,
                 name='rqs',
                 bin_range=[-10.0, 10.0],
                 num_bins=32,
                 hidden_dim=200,
                 kernel_initializer='truncated_normal',
                 **kwargs):
        """
    Creates a SplineBijector layer instance

    Parameters
    ----------
    data_dim : int
        Dimension of the output data. MUST specify becuase input and output
        may not be the same size (will not be if using RealNVP with odd event_shape).
    bin_range : list or tuple, default [-10.0, 10.0]
        Must have exactly 2 elements representing the range of data the flow is applied to.
    num_bins : int, default 32
        Number of bins for spline knots.
    hidden_dim : int, default 200
        Number of hidden dimensions in neural nets.
    """
        super(SplineBijector, self).__init__(name=name, **kwargs)
        self.data_dim = data_dim
        self.bin_min = bin_range[0]
        self.bin_max = bin_range[1]
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        # Create an initial neural net layer
        self.d1 = tf.keras.layers.Dense(self.hidden_dim,
                                        name='d1',
                                        activation=tf.nn.relu,
                                        kernel_initializer=self.kernel_initializer)
        # Create neural nets for widths, heights, and slopes
        self.bin_widths = tf.keras.layers.Dense(self.data_dim * self.num_bins,
                                                activation=None,
                                                name='w',
                                                kernel_initializer=self.kernel_initializer)
        self.bin_heights = tf.keras.layers.Dense(self.data_dim * self.num_bins,
                                                 activation=None,
                                                 name='h',
                                                 kernel_initializer=self.kernel_initializer)
        self.knot_slopes = tf.keras.layers.Dense(self.data_dim * (self.num_bins - 1),
                                                 activation=None,
                                                 name='s',
                                                 kernel_initializer=self.kernel_initializer)

    def call(self, input_tensor, nunits=None):
        """
    Takes an input tensor and returns a RationalQuadraticSpline object.

    The second input, nunits is necessary for compatibility with tfp.bijectors.RealNVP

    Parameters
    ----------
    input_tensor : tf.Tensor
        Input to this layer
    nunits : N/A
        Not used, but required for compatibility with tfp.bijectors.RealNVP

    Returns
    -------
    tfp.bijectors.RationalQuadraticSpline instance
    """
        # Don't use nunits because create nets beforehand
        del nunits

        # Need to check some things based on shape of input
        # First, make sure it has a batch dimension
        in_shape = tf.shape(input_tensor)
        if len(tf.shape(input_tensor)) <= 1:
            input_tensor = tf.reshape(input_tensor, (1, -1))

        # Next, if have event_shape = 1 (transforming 1D distribution), will mask nothing with RealNVP
        # As a result, what is passed will have zero last dimension
        # Can't use data itself to transform itself, so RealNVP just passes something with zero dimension
        # So just learn spline transformations that start with ones as input
        if tf.shape(input_tensor)[-1] == 0:
            input_tensor = tf.ones((tf.shape(input_tensor)[0], 1))

        d1_out = self.d1(input_tensor)
        # Use nets to get spline parameters
        bw = self.bin_widths(d1_out)
        # Apply "activations" manually just to be safe
        bw = self._bin_positions(bw)
        bh = self.bin_heights(d1_out)
        bh = self._bin_positions(bh)
        ks = self.knot_slopes(d1_out)
        ks = self._slopes(ks)

        # If added batch dimension, need to remove now
        if len(in_shape) <= 1:
            bw = tf.squeeze(bw, axis=0)
            bh = tf.squeeze(bh, axis=0)
            ks = tf.squeeze(ks, axis=0)

        return tfp.bijectors.RationalQuadraticSpline(bin_widths=bw,
                                                     bin_heights=bh,
                                                     knot_slopes=ks,
                                                     range_min=self.bin_min)

    def get_config(self):
        config = super(SplineBijector, self).get_config()
        config.update({
            "data_dim": self.data_dim,
            "bin_range": [self.bin_min, self.bin_max],
            "num_bins": self.num_bins,
            "hidden_dim": self.hidden_dim,
            "kernel_initializer": self.kernel_initializer,
        })
        return config


class RQSSplineRealNVP(tf.keras.layers.Layer):
    """
  Abstraction of tfp.bijectors.RealNVP with rational quadratic spline bijector functions.

  Allows for setting up a single flow with multiple blocks.
  Limited to transforming distributions with 1D event shape.

  Attributes
  ----------
  data_dim : int
      Dimensionality of data being transformed.
  num_blocks : int
      Number of spline bijector RealNVP blocks in the chain.
  rqs_params : dict
      Dictionary of keyword arguments for SplineBijector
  batch_norm : bool
      Whether or not batch normalization layers are placed between spline bijectors.
  conditional : bool
      Whether or not conditional inputs are accepted. This is always False for this class.
  """

    def __init__(self, num_blocks=4, rqs_params={}, batch_norm=False, name='rqs_realNVP', **kwargs):
        """
      Creates a RQSSplineRealNVP layer

      Parameters
      ----------
      num_blocks : int, default 4
        Number of RealNVP blocks (data splits, should be at least 2).
      rqs_params : dict, default {}
        Dictionary of keyword arguments for SplineBijector.
      batch_norm : bool, default False
        Whether or not to apply batch normalization between blocks.
    """
        super(RQSSplineRealNVP, self).__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.rqs_params = rqs_params
        self.batch_norm = batch_norm
        self.conditional = False

    def build(self, input_shape):
        self.data_dim = input_shape[-1]

        # Want to create a spline bijector based RealNVP bijector for each block
        # (num_blocks should be at least 2 to be useful, so should add warning at some point)
        # In case data_dim is not even, figure out lengths of split
        block_list = []

        for i in range(self.num_blocks):
            # If data is only length one, all masked and no transform
            # Eventually write warning for this, if not error
            if self.data_dim == 1:
                this_mask = 0
                num_transform = 1
            # Otherwise, mask first half of data on even blocks
            # And mask second half on odd blocks
            # Means the specified output dimension for SplineBijector should be conjugate to masking
            else:
                if i % 2 == 0:
                    this_mask = self.data_dim // 2
                    num_transform = self.data_dim - self.data_dim // 2
                else:
                    this_mask = -(self.data_dim - self.data_dim // 2)
                    num_transform = self.data_dim // 2

            if i != 0 and self.batch_norm:
                block_list.append(tfp.bijectors.BatchNormalization(training=False))

            block_list.append(
                tfp.bijectors.RealNVP(
                    num_masked=this_mask,
                    name='block_%i' % i,
                    bijector_fn=SplineBijector(num_transform, **self.rqs_params),
                ))

        # Put together RealNVP blocks into chained bijector
        # Note that chain operates in reverse order
        self.chain = tfp.bijectors.Chain(block_list[::-1])

    def call(self, inputs, training=False):
        """
      Applies rational-quadratic-spline-based RealNVP chain of flow blocks to input.

      The input may be a sample from a distribution, in which case the transformed sample
      is returned, or a tfp.distributions object, in which case a tfp.distributions.TransformedDistribution
      object is returned.

      Parameters
      ----------
      inputs : tf.Tensor
          An input tensor or tfp.distribution.
      training : bool
          Whether or not training is True or False in BatchNormalization bijectors.

      Returns
      -------
      tf.Tensor or tfp.distributions.TransformedDistribution
    """
        # Need to go through and adjust training variable in BatchNormalization bijector layers
        if self.batch_norm:
            for bij in self.chain.bijectors:
                if isinstance(bij, tfp.bijectors.BatchNormalization):
                    bij.training = training

        # If the input to a bijector is a tfp.distributions object, outputs a TransformedDistribution object
        # If input is tensor, outputs transformed tensor, so works either way as a layer
        if issubclass(type(inputs), tfp.distributions.Distribution):
            return tfp.distributions.TransformedDistribution(inputs, self.chain)
        else:
            return self.chain(inputs)

    def get_config(self):
        config = super(RQSSplineRealNVP, self).get_config()
        config.update({"num_blocks": self.num_blocks, "rqs_params": self.rqs_params, "batch_norm": self.batch_norm})
        return config


class MaskedSplineBijector(tf.keras.layers.Layer):
    """
  A spline bijector layer parametrized by masked autoregressive networks.

  Follows tfp example for using rational quadratic splines (as described in Durkan et al.
  2019) to replace affine transformations (in, say, RealNVP). This should allow more flexible
  transformations with similar cost and should work much better with 1D flows. This version
  uses dense layers that are masked to be autoregressive with conditional inputs optional.
  Such a setup is intended to be used with tfp.bijectors.MaskedAutoregressiveFlow, which
  impacts the structure of arguments to the "call" method.

  Attributes
  ----------
  data_dim : int
      Dimension of the input data, which will also be the dimension of the output.
  bin_min : int
      Minimum value for spline to act on data.
  bin_max : int
      Maximum value for spline to act on data.
  num_bins : int
      Number of spline bins (intervals).
  hidden_dim : int
      Dimensionality of any hidden neural network layers.
  conditional : bool
      Whether or not conditional inputs are accepted.
  conditional_event_shape : tuple
      The shape of conditional inputs, if conditional is True.
  input_order : list or str
      List specifying order of processing DOFs or string describing this behavior.
  """

    def _bin_positions(self, x):
        """
    Defines positions of bins between spline knots as an activation to a dense network's output.
    """
        # Unlike other SplineBijector, don't reshape because AutoregressiveNetwork already does
        # So for those networks, event_shape=data_dim and num_params=num_bins
        out = tf.math.softmax(x, axis=-1)
        out = out * (self.bin_max - self.bin_min - self.num_bins * 1e-2) + 1e-2
        return out

    def _slopes(self, x):
        """
    Defines slopes of splines over each bin.
    Acts as a activation over a dense network's output.
    """
        return tf.math.softplus(x) + 1e-2

    def __init__(
            self,
            name='rqs',
            bin_range=[-10.0, 10.0],
            num_bins=32,
            hidden_dim=200,
            kernel_initializer='truncated_normal',
            conditional=False,
            conditional_event_shape=None,
            input_order='left-to-right',  # May want default to be random? Especially with multiple blocks
            **kwargs):
        """
    Creates a MaskedSplineBijector layer instance

    Parameters
    ----------
    bin_range : list or tuple, default [-10.0, 10.0]
        A length-2 list or tupe specifying the range of data flow is applied to.
    num_bins : int, default 32
        Number of bins for spline knots.
    hidden_dim : int, default 200
        Number of hidden dimensions in neural nets
    conditional : bool, default False
        Whether or not conditional inputs should be included
    conditional_event_shape : int, default None
        Shape of conditional inputs; required if conditional is True.
    input_order : str or list
        Can specify order of DOFs or string corresponding to tfp.bijectors.AutoregressiveNetwork inputs.
    """
        super(MaskedSplineBijector, self).__init__(name=name, **kwargs)
        self.bin_min = bin_range[0]
        self.bin_max = bin_range[1]
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.kernel_initializer = kernel_initializer
        self.conditional = conditional
        self.conditional_event_shape = conditional_event_shape
        self.input_order = input_order

    def build(self, input_shape):
        self.data_dim = input_shape[-1]
        # No initial neural net layer since can add hidden layers if desired
        # Create neural nets for widths, heights, and slopes
        self.bin_widths = tfp.bijectors.AutoregressiveNetwork(
            self.num_bins,
            event_shape=self.data_dim,
            conditional=self.conditional,
            conditional_event_shape=self.conditional_event_shape,
            input_order=self.input_order,
            hidden_units=[
                self.hidden_dim,
            ],
            activation=tf.nn.tanh,  # Only applied to hidden units
            name='w',
            kernel_initializer=self.kernel_initializer)
        self.bin_heights = tfp.bijectors.AutoregressiveNetwork(self.num_bins,
                                                               event_shape=self.data_dim,
                                                               conditional=self.conditional,
                                                               conditional_event_shape=self.conditional_event_shape,
                                                               input_order=self.input_order,
                                                               hidden_units=[
                                                                   self.hidden_dim,
                                                               ],
                                                               activation=tf.nn.tanh,
                                                               name='h',
                                                               kernel_initializer=self.kernel_initializer)
        self.knot_slopes = tfp.bijectors.AutoregressiveNetwork(self.num_bins - 1,
                                                               event_shape=self.data_dim,
                                                               conditional=self.conditional,
                                                               conditional_event_shape=self.conditional_event_shape,
                                                               input_order=self.input_order,
                                                               hidden_units=[
                                                                   self.hidden_dim,
                                                               ],
                                                               activation=tf.nn.tanh,
                                                               name='s',
                                                               kernel_initializer=self.kernel_initializer)

    def call(self, input_tensor, conditional_input=None):
        """
    Generates a rational quadratic spline transformation based on applying masked autoregresive networks.

    Parameters
    ----------
    input_tensor : tf.Tensor
        Inputs to this layer
    conditional_input : tf.Tensor, default None
        Conditional inputs to this layer. Required if conditional attribute set to True.

    Returns
    -------
    tfp.bijectors.RationalQuadraticSpline instance
    """
        # Use nets to get spline parameters
        bw = self.bin_widths(input_tensor, conditional_input=conditional_input)
        # Apply "activations" manually since AutoregressiveNetwork does not apply activation on last layer
        bw = self._bin_positions(bw)
        bh = self.bin_heights(input_tensor, conditional_input=conditional_input)
        bh = self._bin_positions(bh)
        ks = self.knot_slopes(input_tensor, conditional_input=conditional_input)
        ks = self._slopes(ks)
        return tfp.bijectors.RationalQuadraticSpline(bin_widths=bw,
                                                     bin_heights=bh,
                                                     knot_slopes=ks,
                                                     range_min=self.bin_min)

    def get_config(self):
        config = super(SplineBijector, self).get_config()
        config.update({
            "bin_range": [self.bin_min, self.bin_max],
            "num_bins": self.num_bins,
            "hidden_dim": self.hidden_dim,
            "kernel_initializer": self.kernel_initializer,
            "conditional": self.conditional,
            "conditional_event_shape": self.conditional_event_shape,
            "input_order": self.input_order,
        })
        return config


class RQSSplineMAF(tf.keras.layers.Layer):
    """
  Abstraction of tfp.bijectors.MaskedAutoregressiveFlow with masked rational quadratic spline bijector functions.

  Allows for setting up a single flow with multiple blocks.
  Limited to transforming distributions with 1D event shape.

  Attributes
  ----------
  data_dim : int
      Dimensionality of data being transformed.
  num_blocks : int
      Number of spline bijector RealNVP blocks in the chain.
  rqs_params : dict
      Dictionary of keyword arguments for MaskedSplineBijector
  batch_norm : bool
      Whether or not batch normalization layers are placed between spline bijectors.
  conditional : bool
      Whether or not conditional inputs are accepted.
  """

    def __init__(self, num_blocks=2, rqs_params={}, batch_norm=False, name='rqs_MAF', **kwargs):
        """
      Creates RQSSplineMAF layer instance

      Parameters
      ----------
      num_blocks : int, default 2
          Number of RealNVP blocks (data splits, should be at least 2)
      rqs_params : dict, default {}
          Dictionary of keyword arguments for MaskedSplineBijector
      batch_norm : bool, default False
        Whether or not to apply batch normalization between blocks
    """
        super(RQSSplineMAF, self).__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.rqs_params = rqs_params
        self.batch_norm = batch_norm
        self.conditional = rqs_params.get('conditional', False)

    def build(self, input_shape):
        self.data_dim = input_shape[-1]

        # Want to create an MAF bijector with a spline bijector for each block
        block_list = []

        for i in range(self.num_blocks):

            # Set up order to alternate on ends and random on all blocks in between
            # Will ignore if already define specific input_order in self.rqs_params
            if i == 0:
                order = 'right-to-left'
            elif i == (self.num_blocks - 1):
                order = 'left-to-right'
            else:
                order = 'random'

            if i != 0 and self.batch_norm:
                block_list.append(tfp.bijectors.BatchNormalization(training=False))

            if "input_order" in self.rqs_params:
                block_list.append(
                    tfp.bijectors.MaskedAutoregressiveFlow(
                        bijector_fn=MaskedSplineBijector(**self.rqs_params),
                        name='block_%i' % i,
                    ))
            else:
                block_list.append(
                    tfp.bijectors.MaskedAutoregressiveFlow(
                        bijector_fn=MaskedSplineBijector(input_order=order, **self.rqs_params),
                        name='block_%i' % i,
                    ))

        # Put together MAF blocks into chained bijector
        # Note that chain operates in reverse order
        self.chain = tfp.bijectors.Chain(block_list[::-1])

    def call(self, inputs, training=False, conditional_input=None):
        """
      Applies a chain of rational-quadratic-spline-based flows parametrized by masked autoregressive networks.

      The input may be a sample from a distribution, in which case the transformed sample
      is returned, or a tfp.distributions object, in which case a tfp.distributions.TransformedDistribution
      object is returned.

      Parameters
      ----------
      inputs : tf.Tensor
          An input tensor or tfp.distribution.
      training : bool, default False
          Whether or not training is True or False in BatchNormalization bijectors.
      conditional_input : tf.Tensor, default None
          Conditional input for the MAF bijectors

      Returns
      -------
      tf.Tensor or tfp.distributions.TransformedDistribution
    """
        # Need to go through and adjust training variable in BatchNormalization bijector layers
        # Also collect names of blocks (i.e., not BatchNormalization layers) to pass conditional inputs to
        # To pass conditional inputs along to MAF bijectors, need dictionary
        # keys will be block names and values will be dictionary with value of conditional_input
        cond_dict = {}
        for bij in self.chain.bijectors:
            if isinstance(bij, tfp.bijectors.BatchNormalization):
                bij.training = training
            else:
                cond_dict[bij.name] = {'conditional_input': conditional_input}

        # If the input to a bijector is a tfp.distributions object, outputs a TransformedDistribution object
        # If input is tensor, outputs transformed tensor, so works either way as a layer
        if issubclass(type(inputs), tfp.distributions.Distribution):
            # Note that defining a kwargs_split_fn to ensure that conditional inputs are passed appropriately
            # If no keyword argument 'bijector_kwargs' is passed, uses cond_dict to pass to bijector
            # Just make sure to provide conditional input when creating a distribution!
            return tfp.distributions.TransformedDistribution(
                inputs,
                self.chain,
                kwargs_split_fn=lambda kwargs:
                (kwargs.get('distribution_kwargs', {}), kwargs.get('bijector_kwargs', cond_dict)))
        else:
            return self.chain(inputs, **cond_dict)

    def get_config(self):
        config = super(RQSSplineMAF, self).get_config()
        config.update({"num_blocks": self.num_blocks, "rqs_params": self.rqs_params, "batch_norm": self.batch_norm})
        return config


class FlowModel(tf.keras.Model):
    """
  Model wrapper around flow for easier, faster training.

  No loss function is necessary, as the computation of loss is built into this layer's call.
  This only makes sense if training a flow to only transform from one distribution to another
  and that is the only objective. Specifically, if you have samples from a distribution you
  want to learn how to generate so that the loss is the log-probability of those samples
  under the flowed distribution (latent distribution passed through the flow).

  Attributes
  ----------
  flow : tfp.bijector object or layer applying a bijector
      Flow that is applied to transform the latent distribution.
  latent_dist : tfp.distributions object
      Distribution that flow acts on.
  """

    def __init__(self, flow, latent_dist, name='flow_model', **kwargs):
        """
    Creates a FlowModel model instance

    Parameters
    ----------
    flow : tfp.bijector object or layer applying bijector
        Flow that will be applied in order to transform latent distribution.
    latent_dist : tfp.distributions object
        Distribution that flow acts upon in the forward direction to produce samples.
    """
        super(FlowModel, self).__init__(name=name, **kwargs)
        self.flow = flow
        self.latent_dist = latent_dist
        # self.layer_dist = tfp.layers.DistributionLambda(lambda x, training=False: self.flow(self.latent_dist,
        #                                                                                     training=training))

    def call(self, inputs, training=False):
        """
    Applies a flow to the latent distribution and calculates log-probability of provided samples.

    Parameters
    ----------
    inputs : tf.Tensor
        Samples from the distribution we want to learn (forward application of the flow
        produces this distribution).
    training : bool, default False
        Whether or not we are training. May be important if have block normalizations in flow layer.

    Returns
    -------
    dist : tfp.distributions.TransformedDistribution instance
    """
        # Start by transforming our latent distribution with our flow
        # Note that flow behavior depends on training boolean
        dist = self.flow(self.latent_dist, training=training)
        # dist = self.layer_dist(inputs, training=training)

        # Since cannot currently see use of having separate loss function, handling here
        # First compute log-probability, then take negative of its mean
        log_p = dist.log_prob(inputs)
        neg_mean_log_p = tf.reduce_mean(-log_p)

        # Add to loss
        # Can incorporate weights on the data, or more advanced losses, like training with KL
        # in both directions
        # To do this, replace compute_loss, which takes (x, y_true, y_pred, sample_weight)
        # Overwriting that would allow for the dual KL loss pretty easily if also have function
        # to compute potential energies
        # That is, you can easily compute the reverse KL loss given y_true and y_pred being the
        # transformed distribution
        # But for the reverse KL, you have to sample from the latent distribution, flow, and
        # calculate potential energies
        self.add_loss(neg_mean_log_p)

        # Return the distribution object, though, not the loss
        return dist

    def predict_step(self, lz):
        """
    A custom predict step makes the model more useful.

    fit and evaluate will compute losses (and metrics) with training set to True and False, respectively.
    The returned loss will typically be the average negative log-probability over the batch.
    predict instead just implements the flow.
    You can always access whatever you need based on model.flow and model.latent_dist, or calling the model..
    But this is a short-cut.

    Parameters
    ----------
    lz : tf.Tensor
        A latent distribution sample.

    Returns
    -------
    z : tf.Tensor
        A transformed sample under flow.
    """
        return self.flow(lz, training=False)

    def get_config(self):
        config = super(FlowModel, self).get_config()
        config.update({
            "flow": self.flow,
            "latent_dist": self.latent_dist,
        })
        return config


# Still need to really figure out how BatchNormalization works in Chain bijector...
# May need to get fancy with how the keyword "training" is passed to different layers...
# At first, just see if it naively works, but test it carefully
# If it doesn't work, need to use bijector_kwargs={'training": True} when calling dist.log_prob
# May not work though, and may need to specify keywords for BatchNormalization blocks by name...
# Do that by naming the inner bijectors and passing keyword arguments as their names with dictionaries, i.e.,
# flow.forward(lz, b1={'training': False}, b2={'training': False})
# That seems pretty impractical, though
# May just be easier to implement a DistributionLambda layer, which may automatically
# handle passing training=True/False
# That would simplify training, but still make it hard to use the transformed distribution to
# compute log-probabilities
# (i.e., essentially in evaluate mode, where want to compute the loss with training=False)
# As far as I can tell, it would not impact the forward pass, so wouldn't matter for flow transformations or sampling
# That's because only inverse has different behavior with training set to True or False
# ACTUALLY...
# The thing to do is probably to rebuild the transformed distribution every time depending on
# whether training is True or False
# i.e., should set up make_distribution_fn in DistributionLambda so that behaves differently
# Only issue there is that make_distribution_fn only takes the "arguments" of the last layer...
# So should only passing training to layer_dist, not actual input...
# Then can set training to True or False in BatchNormalization layers!
# However, that breaks the whole nice object-oriented structure, where then we can't just pass a
# created flow object (Chain bijector)

# Seems like resolved issues above by passing function to create flow bijector to Model
# It's call method takes "training" as an input argument so can create a bijector with different behavior
# However, that creates new networks with each call...
# Really confused on the whole BatchNormalization bijector thing, so need to investigate

# I believe it is figured out now.
# All flows are now tf.keras.layers.Layer objects
# That allows them to have call methods and be serializable for easy saving.
# You then just apply the flow, passing training to it's call method, in the call of the Model
# The Model, when called this way, will output the transformed distribution with the flow applied
# In tfp, by default this is a TransformedDistribution object if the input is a tfp.distributions object
# The only weird thing is that you do actually have to manually change the "training" variable in
# each BatchNormalization bijector
# It seems like that should be taken care of somehow with keras...
# This should be tested.
# However, either way the current setup with flows as keras layers should encourage correct behavior

# class FlowLossForward(tf.keras.losses.Loss):
#  """
#  Loss function for typical, basic flow structure, only using samples from target distribution for training.
#  """
#
#  def call(self, z, dist):
#    """
#    Inputs:
#        z - samples from target distribution
#        dist - transformed distribution (tfp.distributions object) from flow, calling FlowModel
#    Outputs:
#        loss - negative log-probability of samples under flowed distribution
#               (note that returns each element in batch, so want to apply mean reduction when training)
#    """
#    return -dist.log_prob(z)

# Applying a loss with the dual loss (forward and reverse KL) is not so easy...
# This is because it doesn't fit into the loss signature (y_true, y_pred)
# You would need a custom model to do the dual, so it may make sense to just to add_loss in the model instead
# Unless there is a compelling reason for why a different loss might be useful, which I do not see right now
# At least with distribution object, the log-probablity is mostly what makes sense.
