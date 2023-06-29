"""
Defines tf.keras.layers.Layer subclasses that represent probability distributions.

Specifically, when these layers are called with provided inputs, the output is a
tensorflow-probability distribution class object.
In the simplest cases, these are extensions of tensorflow-probability layers, such
as a layer representing independent von Mises distributions. In some cases, however,
the core layer class of the layers module of tensorflow-probability, DistributionLambda,
cannot be subclassed, and so custom layers have been created for those cases. This
includes wrapping independent Blockwise distributions, as well as Autoregressive
and flowed distributions.
"""

# Mostly creates layers that represent probability distributions
# If just want distributions, can do that with tfp
# Or by creating one of these layers and calling it
# Essentially, for something like a prior, can just create a tfp distribution object
# (for one with a normalizing flow, use the bijectors or flow layers in flows.py, BUT
# if you want multiple blocks with batch normalization anywhere, you will actually need
# a function to GENERATE TransformedDistribution objects with training on the batch norm
# blocks set the right way, so will want to use one of the layers in flows.py for sure)

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import distribution_util as dist_util


# This probably should not be a class...
# Really need a function to generate the transformations and that's it,
# which is essentially the __init__ method of this class.
# Making this a class just complicates things since we are really truly just wrapping a function
# that we create.
def make_param_transform(dist_class=None, transform_fn=tf.identity):
    """
  A function to create functions that transform inputs into parameters on a domain appropriate to a distribution.

  By default, a tfp.distribution class, if provided, is used to construct the transformation, with an exception
  for von Mises already applied so that 2 parameters are taken for the location and treated as a
  sine-cosine pair that is transformed into the domain [-np.pi, np.pi], which helps training.
  If the default transformations in the parameter_properties attribute of the distribution class
  are not what is desired, you can instead pass an explicit callable (likely a lambda function)
  to transform to parameters. If neither a distribution class nor a callable is provided,
  the default behavior is to set the transformation function to tf.identity.
  The output of a call to an object of this class should be a dictionary with the keys being the
  names of distribution parameters and the values being the transformed values.
  The input should just be a list, array, tensor, etc. of the untransformed values.

  Parameters
  ----------
  dist_class : tfp.distributions class, default None
      A class within tfp.distributions with the attribute parameter_properties
  transform_fn : callable, default tf.identity
      A callable to define the transformation

  Returns
  -------
  param_transform : function
      The callable transformation function
  """
    if dist_class is not None:
        props_dict = dist_class.parameter_properties()

        # Define in special way for von Mises to wrap into periodic interval
        if dist_class.__name__ == 'VonMises':

            def _fn(x):
                return {
                    'loc': props_dict['loc'].default_constraining_bijector_fn()(tf.math.atan2(x[..., 0], x[..., 1])),
                    'concentration': props_dict['concentration'].default_constraining_bijector_fn()(x[..., 2])
                }

        else:

            def _fn(x):
                return {
                    k: props_dict[k].default_constraining_bijector_fn()(x[..., i])
                    for (i, k) in enumerate(props_dict.keys())
                }

        param_transform = _fn

    else:
        param_transform = transform_fn

    return param_transform


# Turns out inheriting from DistributionLambda does not work for autoregression or flows
# Could make it work for both of those, but only if have no conditional inputs
# So not much point in keeping IndependentBlockwise as a DistributionLambda inherited class...
# At some point, should just make tf.keras.layers.Layer so can inherit it into AutoregressiveBlockwise
# and avoid some code repetition


class IndependentBlockwise(tf.keras.layers.Layer):
    """
  A layer to create a Blockwise distribution of independent, arbitrary tfp.distribution classes.

  This class follows the style of other independent distribution layers in tfp.
  However, it inherits directly from tf.keras.layers.Layer for simplicity.
  """

    def __init__(
        self,
        num_dofs,
        dist_classes,  # Consider helper functions to create list from topology in future... for now manual
        param_nums=None,
        param_transforms=None,
        name='independent_blockwise',
        **kwargs,
    ):
        """
    Creates IndependentBlockwise layer

    Parameters
    ----------
    num_dofs : int
        Number of degrees of freedom or dimensionality of distribution (i.e., event_shape)
    dist_classes : tfp.distribution class or list
        List of tfp.distribution classes for probability on each degree of freedom.
        If just a single distribution class, it will be used for all dofs.
    param_nums : int or list of ints, default None
        List of integers specifying the number of parameters to pass to each distribution
        in dist_classes for its creation; if a single integer, uses for all distributions.
        Necessary because one cannot ALWAYS infer the number of parameters from the distribution.
        If left as None, the default, the number of parameters will be inferred from the classes
        in dist_classes.
    param_transforms : callable or list of callables, default None
        Should be a callable function or list of callable functions with a single callable function
        for each dof that expects to transform param_nums[i] inputs for the ith dof distribution.
        If left as None (the default), will infer transformations from dist_classes using the
        parameter_properties attribute.

    Returns
    -------
    IndependentBlockwise distribution layer instance
    """
        super(IndependentBlockwise, self).__init__(name=name, **kwargs)

        self.num_dofs = num_dofs

        # Check and specify dist_classes
        if not isinstance(dist_classes, (list, tuple)):
            if not issubclass(dist_classes, tfp.distributions.Distribution):
                raise TypeError("Expected a tfp.distributions class object, but got %s" % type(dist_classes).__name__)
            self.dist_classes = [dist_classes] * self.num_dofs
        else:
            if len(dist_classes) != self.num_dofs:
                raise ValueError(
                    "If specifying a list of tfp.distributions objects, must be of same length as number of degrees \
                    of freedom (%i) but got %i." % (self.num_dofs, len(dist_classes)))
            self.dist_classes = dist_classes

        # Check and specify param_nums
        if param_nums is None:
            param_nums = []
            for d in self.dist_classes:
                # Only collect parameters if "preferred," which removes duplicates
                this_num = sum(v.is_preferred for (k, v) in d.parameter_properties().items())
                # Add extra for von Mises distribution so can ensure location is in periodic domain
                if d.__name__ == 'VonMises':
                    this_num += 1
                param_nums.append(this_num)
            self.param_nums = param_nums
        else:
            if not isinstance(param_nums, (list, tuple)):
                self.param_nums = [param_nums] * self.num_dofs
            else:
                if len(param_nums) != self.num_dofs:
                    raise ValueError(
                        "If specifying a list of parameter numbers, must be of same length as number of degrees of \
                        freedom (%i) but got %i." % (self.num_dofs, len(param_nums)))
                self.param_nums = param_nums

        # Check and specify param_transforms
        if param_transforms is None:
            self.param_transforms = [make_param_transform(dist_class=d) for d in self.dist_classes]
        else:
            if not isinstance(param_transforms, (list, tuple)):
                self.param_transforms = [tf.identity] * self.num_dofs
            else:
                if len(param_transforms) != self.num_dofs:
                    raise ValueError(
                        "If specifying a list of parameter transformations, must be of same length as number of \
                        degrees of freedom (%i) but got %i." % (self.num_dofs, len(param_transforms)))
                self.param_transforms = param_transforms

    def call(self, inputs):
        """
    Creates a Blockwise distribution from an input tensor.

    Parameters
    ----------
    inputs : tf.Tensor
        The input to this layer

    Returns
    -------
    tfp.distributions.Blockwise class instance
    """
        params = tf.split(inputs, self.param_nums, axis=-1)
        # Loop over distribution classes and parameters, creating list of distribution objects
        # Seems like should be something better for tensorflow than looping and appending to a list...
        dist_list = []
        for i in range(self.num_dofs):
            dist_list.append(self.dist_classes[i](**self.param_transforms[i](params[i])))

        return tfp.distributions.Blockwise(dist_list)

    def params_size(self):
        """
    Outputs the total number of parameters,

    This is useful for specifying the output of layers before this one.

    Parameters
    ----------
    None

    Returns
    -------
    Total number of parameters
    """
        return sum(self.param_nums)

    def get_config(self):
        config = super(IndependentBlockwise, self).get_config()
        config.update({
            "num_dofs": self.num_dofs,
            "dist_classes": self.dist_classes,
            "param_nums": self.param_nums,
            "param_transforms": self.param_transforms,
        })
        return config


class AutoregressiveBlockwise(IndependentBlockwise):
    """
  A layer based on an autoregressive distribution composed through a blockwise distribution.

  Specifically, this is a tfp.distribution.Autoregressive distribution whose distribution_fn argument
  is a function that produces a tfp.distribution.Blockwise distribution instance.
  Note that we cannot inherit from DistributionLambda - must inherit from keras Layer directly.
  Doing this allows for owning of the autoregressive network parameters, and for conditional inputs.
  Note that means this layer ONLY produces a distribution object - for sampling, must call sample from distribution.
  """

    def __init__(
        self,
        *args,
        conditional=False,
        conditional_event_shape=None,
        name='autoregressive_blockwise',
        **kwargs,
    ):
        """
    Creates AutoregressiveBlockise layer

    See IndependentBlockwise for most arguments.

    Parameters
    ----------
    conditional : bool, default False
        Whether or not to use conditional inputs in autoregressive network
    conditional_event_shape : int or tuple, default None
        Shape of conditional inputs (excluding batch dimensions). Necessary if conditional is True.

    Returns
    -------
    AutoregressiveBlockwise distribution layer instance
    """
        super(AutoregressiveBlockwise, self).__init__(*args, name=name, **kwargs)

        self.conditional = conditional
        self.conditional_event_shape = conditional_event_shape

    def build(self, input_shape):
        # Nice to have build method here so can create own autoregressive network
        if input_shape[-2:] != (self.num_dofs, max(self.param_nums)):
            raise ValueError(
                "Last (assuming only non-batch) dimension is of size %s, but must match number of specified degrees \
                of freedom, %s." % (str(input_shape[-2:]), str((self.num_dofs, max(self.param_nums)))))

        # Create autoregressive network
        # Using maximum number of parameters any distribution will have, which will generally create too many
        # However, will just slice to take what we need
        self.auto_net = tfp.bijectors.AutoregressiveNetwork(max(self.param_nums),
                                                            self.num_dofs,
                                                            conditional=self.conditional,
                                                            conditional_event_shape=self.conditional_event_shape)

    def call(self, inputs, conditional_input=None):
        """
    Creates and returns an Autoregressive blockwise distribution from an input tensor

    Parameters
    ----------
    inputs : tf.Tensor
        The input to this layer
    conditional_input : tf.Tensor
        Conditional input

    Returns
    -------
    tfp.distributions.Autoregressive instance
    """

        # First, need to create a function that creates our distribution
        # Need to do that in here so that can pass conditional inputs along
        # Necessary since in AutoregressiveDistribution, make_distribution_fn only takes
        # samples from previous call and distribution.sample(), so nowhere to specify
        # conditional inputs
        def _make_dist(samples):
            # Will treat parameters as inputs plus shift due to autoregressive network
            raw_params = inputs + self.auto_net(samples, conditional_input=conditional_input)
            params = tf.unstack(raw_params, axis=-2)
            # Loop over distribution classes and parameters, creating list of distribution objects
            dist_list = []
            for i in range(self.num_dofs):
                dist_list.append(self.dist_classes[i](**self.param_transforms[i](params[i])))
            return tfp.distributions.Blockwise(dist_list)

        return tfp.distributions.Autoregressive(_make_dist,
                                                sample0=tf.ones(tf.shape(inputs)[:-1]),
                                                num_steps=self.num_dofs)

    def params_size(self):
        """
    Returns the shape of the autoregressive netowrk output.

    This should be the output size of a layer before this one.
    This is the number of dofs by the number of parameters.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        (N_dofs, N_params)
    """
        return (self.num_dofs, max(self.param_nums))

    def get_config(self):
        config = super(AutoregressiveBlockwise, self).get_config()
        config.update({
            "conditional": self.conditional,
            "conditional_event_shape": self.conditional_event_shape,
        })
        return config


class FlowedDistribution(tf.keras.layers.Layer):
    """
  A layer that creates a TransformedDistribution based on a flow.

  Essentially just wraps a distribution layer and a flow layer into one for convenience.
  Unfortunately, we cannot inherit from DistributionLambda because within a tf.keras.layers.Lambda
  layer, it is not possible to dynamically pass keyword arguments, which prevents us from
  passing conditional_input (applicable to an MAF bijector).
  As such, this ONLY produces a distribution, with no concretization.
  That means that sampling must be done manually within a custom Model if that is necessary.
  """

    def __init__(self, flow, latent_dist, name='flowed_dist', **kwargs):
        """
    Creates a FlowedDistribution class instance.

    Parameters
    ----------
    flow : tf keras layer
        A layer that, in its call method, takes a distribution as input and outputs a
        tfp.distributions.TransformedDistribution object by applying a bijector.
        Intended to utilize one of the flows in vaemolsim.flows, such as RQSSplineRealNVP
        or RQSSplineMAF.
    latent_dist : tfp.layers-like instance
        A layer representing a distribution to be transformed (i.e., a layer that, given
        inputs, produces a tfp.distributions object)

    Returns
    -------
    FlowedDistribution layer instance
    """
        super(FlowedDistribution, self).__init__(name=name, **kwargs)

        self.flow = flow
        self.latent_dist = latent_dist
        self.conditional = self.flow.conditional

    def call(self, inputs, training=False, **kwargs):
        """
    Produces a tfp.distributions.TransformedDistribution object from inputs.

    Applies transformation by creating starting distribution given inputs (from last layer), then
    using flow to produce a TransformedDistribution object.
    Note that kwargs is accepted and will be passed on to the flow to allow passing conditional_input.
    It is not hard-coded as conditional input, though, because some flows, like RealNVP do not take
    conditional_input as an argument.

    Parameters
    ----------
    inputs : tf.Tensor
        Input to this layer
    training : bool, default False
        Whether or not training or making predictions; applicable if have batch normalization layers
    **kwargs : other keyword arguments
        Necessary to capture conditional_input if provided and pass to flow

    Returns
    -------
    tfp.distributions.TransformedDistribution instance
    """
        start_dist = self.latent_dist(inputs)
        # Passing kwargs necessary to capture conditional_input if provided
        return self.flow(start_dist, training=training, **kwargs)

    def params_size(self):
        """
    Returns parameter size for distribution layer feeding into the flow.

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        Size of parameters needed for inputs to the distribution to be transformed
    """
        return self.latent_dist.params_size()

    def get_config(self):
        config = super(FlowedDistribution, self).get_config()
        config.update({
            "flow": self.flow,
            "latent_dist": self.latent_dist,
        })
        return config


# Could instead consider just creating a StaticDistribution layer
# This would just return the same tfp distribution object regardless of inputs
# Could the wrap that in FlowedDistribution without requiring StaticFlowedDistribution
# Both are equivalent, just packaged differently
class StaticFlowedDistribution(tf.keras.layers.Layer):
    """
  Similar to FlowedDistribution, but takes a static tfp.distributions object as the starting point.

  Using this to produce TransformedDistribution objects rather than
  just creating a TransformedDistribution object allows for bijectors (flows) that
  have different behavior with/without training, as is the case when batch norm
  bijectors are added between flow blocks. This is mainly relevant to creating prior
  distributions.
  """

    def __init__(self, flow, latent_dist, name='static_flowed_dist', **kwargs):
        """
    Creates a StaticFlowedDistribution layer

    Parameters
    ----------
    flow : tf keras layer
        A layer that, in its call method, takes a distribution as input and outputs a
        tfp.distributions.TransformedDistribution object by applying a bijector.
        Intended to utilize one of the flows in vaemolsim.flows, such as RQSSplineRealNVP or RQSSplineMAF
    latent_dist : tfp.distribution
        A tfp.distribution object representing a distribution to be transformed

    Returns
    -------
    StaticFlowedDistribution layer instance
    """
        super(StaticFlowedDistribution, self).__init__(name=name, **kwargs)

        self.flow = flow
        self.latent_dist = latent_dist

    def __call__(self, inputs, training=False):
        """
    Produces a tfp.distributions.TransformedDistribution object.

    Note that this function will ignore inputs since latent distribution is static. It does, however
    take a training keyword argument that will be passed to the flow.

    Parameters
    ----------
    inputs : N/A
        Will be ignored
    training : bool, default False
        Whether or not we are training or making predictions.
        Applicable if have block normalization layers in flow.
    """
        # Passing kwargs necessary to capture conditional_input if provided
        return self.flow(self.latent_dist, training=training)

    def get_config(self):
        config = super(StaticFlowedDistribution, self).get_config()
        config.update({
            "flow": self.flow,
            "latent_dist": self.latent_dist,
        })
        return config


# For convenience (because will likely use it often for encoders), create IndependentVonMises layer
# Just copying tfp.layes.IndependentNormal... turns out there are nice things about the whole tfp layer thing
# It is easier to build because you don't have to pass a sample through the flow because the "shape" of the
# distribution object as a _TensorCoercible is the event_shape, which can be used by the flow to build
class IndependentVonMises(tfp.layers.DistributionLambda):
    """An independent von Mises Keras layer.

  Note that, though the von Mises distribution has 2 parameters, this layer expects 3 inputs.
  That is because atan2 will be applied to the first two arguments to interpret them as a
  sine-cosine pair for the location, which neatly wraps it into the domain [-pi, pi].
  This should help avoid degeneracies during training.

  """

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfp.distributions.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        """Initialize the `VonMisesNormal` layer.

    Args:
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
        convert_to_tensor_fn = tfp.layers.distribution_layer._get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop('make_distribution_fn', None)

        super(IndependentVonMises, self).__init__(lambda t: IndependentVonMises.new(t, event_shape, validate_args),
                                                  convert_to_tensor_fn, **kwargs)

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentVonMises'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(event_shape,
                                                                          name='event_shape',
                                                                          dtype_hint=tf.int32),
                                                     tensor_name='event_shape')
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape,
            ], axis=0)
            sine_params, cosine_params, scale_params = tf.split(params, 3, axis=-1)
            loc_params = tf.atan2(sine_params, cosine_params)
            return tfp.distributions.Independent(tfp.distributions.VonMises(loc=tf.reshape(loc_params, output_shape),
                                                                            concentration=tf.math.softplus(
                                                                                tf.reshape(scale_params,
                                                                                           output_shape)),
                                                                            validate_args=validate_args),
                                                 reinterpreted_batch_ndims=tf.size(event_shape),
                                                 validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentVonMises_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(3) * tfp.layers.distribution_layer._event_size(
                event_shape, name=name or 'IndependentVonMises_params_size')

    def get_config(self):
        """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': tfp.layers.distribution_layer._serialize(self._convert_to_tensor_fn),
            'validate_args': self._validate_args
        }
        base_config = super(IndependentVonMises, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Will be useful to have this as well
class IndependentDeterministic(tfp.layers.DistributionLambda):
    """An independent deterministic Keras layer.

  Just takes the outputs of the last layer (typically the encoder mapping), and creates a
  deterministic distribution where sampling reproduces the inputs. The existence of this
  class is not utilitarian since there is nothing really that you can do with this more
  than what you could do with the values input to create this distribution. However,
  it is mathematically consistent with use of other distributions.
  """

    def __init__(self,
                 event_shape=(),
                 convert_to_tensor_fn=tfp.distributions.Distribution.sample,
                 validate_args=False,
                 **kwargs):
        """Initialize the `Deterministic` layer.

    Args:
      event_shape: integer vector `Tensor` representing the shape of single
        draw from this distribution.
      convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
        instance and returns a `tf.Tensor`-like object.
        Default value: `tfd.Distribution.sample`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
    """
        convert_to_tensor_fn = tfp.layers.distribution_layer._get_convert_to_tensor_fn(convert_to_tensor_fn)

        # If there is a 'make_distribution_fn' keyword argument (e.g., because we
        # are being called from a `from_config` method), remove it.  We pass the
        # distribution function to `DistributionLambda.__init__` below as the first
        # positional argument.
        kwargs.pop('make_distribution_fn', None)

        super(IndependentDeterministic,
              self).__init__(lambda t: IndependentDeterministic.new(t, event_shape, validate_args),
                             convert_to_tensor_fn, **kwargs)

        self._event_shape = event_shape
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_shape=(), validate_args=False, name=None):
        """Create the distribution instance from a `params` vector."""
        with tf.name_scope(name or 'IndependentDeterministic'):
            params = tf.convert_to_tensor(params, name='params')
            event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(event_shape,
                                                                          name='event_shape',
                                                                          dtype_hint=tf.int32),
                                                     tensor_name='event_shape')
            output_shape = tf.concat([
                tf.shape(params)[:-1],
                event_shape,
            ], axis=0)
            return tfp.distributions.Independent(tfp.distributions.Deterministic(loc=tf.reshape(params, output_shape),
                                                                                 validate_args=validate_args),
                                                 reinterpreted_batch_ndims=tf.size(event_shape),
                                                 validate_args=validate_args)

    @staticmethod
    def params_size(event_shape=(), name=None):
        """The number of `params` needed to create a single distribution."""
        with tf.name_scope(name or 'IndependentDeterministic_params_size'):
            event_shape = tf.convert_to_tensor(event_shape, name='event_shape', dtype_hint=tf.int32)
            return np.int32(1) * tfp.layers.distribution_layer._event_size(
                event_shape, name=name or 'IndependentDeterministic_params_size')

    def get_config(self):
        """Returns the config of this layer.

    NOTE: At the moment, this configuration can only be serialized if the
    Layer's `convert_to_tensor_fn` is a serializable Keras object (i.e.,
    implements `get_config`) or one of the standard values:
     - `Distribution.sample` (or `"sample"`)
     - `Distribution.mean` (or `"mean"`)
     - `Distribution.mode` (or `"mode"`)
     - `Distribution.stddev` (or `"stddev"`)
     - `Distribution.variance` (or `"variance"`)
    """
        config = {
            'event_shape': self._event_shape,
            'convert_to_tensor_fn': tfp.layers.distribution_layer._serialize(self._convert_to_tensor_fn),
            'validate_args': self._validate_args
        }
        base_config = super(IndependentDeterministic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# STILL WORKING ON THIS ONE!
# It's complicated because of the need to update the FG coordinates supplied
# to the distance mask and attention
# We want to create the distributions for each set of FG degrees of freedom
# within the local coordinate system of the CG site
# That makes the probabilistic modeling easier and more tightly coupled to the physics
# (can use internal DOFs, etc.)
# BUT, for the masking and geometric algebra attention, we need Cartesian coordinates
# in the global frame
# That implies an additional conversion function INSIDE the _make_dist_fn
# And this would need to take both the CG coords and the local coords and act on each pair...
# But then each time you call _make_dist_fn, you're having to recompute that mapping to
# global Cartesian coords for ALL previous
# Is it possible to just keep a tensor of FG Cartesian coords owned by the layer and have
# each _make_dist_fn slice over that?
# Should be, just have to make sure it takes the right slice... in other words, it has to mask it correctly
# So maybe just update a new mask for each _make_dist_fun?
# Lots of options, with one shown below, though that requires lots of information about each residue, etc.
# And the one shown is where you compute global coords for all previously decoded residues rather
# than stashing them and slicing
class JointDistribution(tf.keras.layers.Layer):
    """
  WORK IN PROGRESS

  A distribution composed of a series of distributions based on a repeatedly applied function that
  takes the outputs of all previous distributions and uses them to parametrize the next distribution.

  Inputs to layer are CG coordinates
  In call, must CREATE function to produce each distribution in turn.
  Then pass that, along with the distribution for the first DOFs, to a JointDistributionSequentialAutoBatched object
  """

    def __init__(self, res_names, dist_dict, coord_map_dict, atom_props_dict, name='joint_dist', **kwargs):
        """
    Inputs:
        res_names - list of residue/molecule names for each residue/molecule (should correspond to names in dist_dict)
        dist_dict - a dictionary mapping names of residues/molecules to distribution-making objects
        coord_map_dict - a dictionary mapping names of residues/molecules to mappings from a CG site origin to
                         rectangular atomic coords; the callable for the mapping should take as arguments the
                         local (internal) coordinates and the CG coordinate
        atom_props_dict - a dictionary mapping residue/molecule names to properties of all atoms that compose them;
                          this would be for input to something like an SchNet over atom locations and properties
    Outputs:
        JointDistribution layer
    """
        super(JointDistribution, self).__init__(name=name, **kwargs)
        self.res_names = res_names
        self.dist_dict = dist_dict
        self.coord_map_dict = coord_map_dict
        self.props_dict = atom_props_dict

    def call(self, inputs, training=False):
        # Treat each input as coordinates of a CG site
        # (could be just 3 rectangular coords, or backbone plus centroid, etc.)
        # Or at least it's much harder with proteins... probably actually just want
        # CG coords but exclude some backbone atoms somehow
        # Just need to make sure the atomic coords you decode to are internal (BAT)
        # coords for JUST sidechain atoms
        # So for alpha carbon, just add a hydrogen (predict two angles, with mapping handling bond length)
        # For beta carbon, predict two (or more if alanine) hydrogen positions as angles and/or dihedrals
        # For others, predict bonds, angles, dihedrals, etc. as needed with alpha carbon
        # as root atom and ignoring beta carbon coords

        # Each function created to make a distribution will have a different reference CG site
        # Or think of as needing function to take CG sites and produce (N_batch, N_cg, N_descriptors) tensor
        # (likely a SchNet type thing - actually, geometric algebra attention)
        # Each of the entries along axis 1 are the fixed inputs to a function to produce a distribution,
        # of shape (N_batch, N_descriptors)
        # That way they are of fixed size and should contain info about the nearest neighbor CG
        # sites and their properties
        descriptors = cgschnet(
            inputs, self.res_names)  # Or pass property list instead of names? Something like LJ params, charges, etc.
        # Because will have to pass atom properties, can have local CG attention network same as FG
        # (just different atom types)
        # Could also make separate, but seems easier to just concatenate all coordinates and properties together
        # As an extra bonus, concatenation will let CG and FG coordinates attend to each other,
        # which may improve performance

        # Some way to create random permutation/decoding order?  Probably do when encoding, actually,
        # then can sort out after decoding.
        # Do in overall model!
        # And make sure when do it, permute res_names attribute in this layer (then unpermute at end in model)

        # Create empty list to hold distribution-creating functions
        dist_list = []

        # Add first distribution for first CG dof to list
        # Note that distribution creating callables in dist_dict should have mapping to convert descriptors to
        # shape they need
        # So maybe should actually be MappingToDistribution objects
        dist_list.append(self.dist_dict[self.res_names[0]](descriptors[:, 0, :],
                                                           atom_descriptors=None,
                                                           training=training))

        # Loop over other CG sites and add functions
        for i in range(1, len(self.res_names)):

            # Arguments are flexible so will take any number (all previous samples)
            def _make_dist_fn(*args):
                # First want to convert all previously decoded coordinates (arguments) to rectangular coordinates
                # Each item in list should be the residue/molecule local atomic coordinates with its CG site
                # as the origin
                # So loop over args to convert
                # Also need to collect properties of atoms somehow to improve SchNet
                atom_coords = []
                atom_props = []
                for j, a in enumerate(args):
                    atom_coords.append(self.coord_map_dict[self.res_names[i - (j + 1)]](a, inputs[:, i - (j + 1), :]))
                    atom_props.append(self.props_dict[self.res_names[i - (j + 1)]])
                # Concatenate decoded global rectangular coordinates
                # Do over axis 1 because should be (N_batch, N_CG_decoded, 3)
                # May also need to concatenate atom_props, too, but figuring that out with SchNet
                atom_coords = tf.concat(atom_coords, axis=1)
                # Apply an schnet to the decoded coordinates
                # (may just be way to collect nearby coordinates and convert to local coords)
                atom_descriptors = atomschnet(inputs[:, i, :], atom_coords, atom_props)
                # Return distribution using appropriate distribution-creating function
                return self.dist_dict[self.res_names[i]](descriptors[:, i, :],
                                                         atom_descriptors=atom_descriptors,
                                                         training=training)

            dist_list.append(_make_dist_fn)

        return tfp.distributions.JointDistributionSequentialAutoBatched(dist_list, batch_ndims=1)
