"""
Defines layers that perform coordinate transformations or mappings between data representations.

All classes subclass tf.keras.layers.Layer. In the simplest case, the mapping
is just a fully-connected, deep neural network, though added features are
presented to handle periodic degrees of freedom, which arise naturally in
molecular systems. Other examples include dealing with mappings to
coarse-grained coordinates, as well as applying distance-based masks and
subsequently local embeddings based on geometric algebra.
"""

import numpy as np
import tensorflow as tf

from geometric_algebra_attention import keras as gaa_keras


class FCDeepNN(tf.keras.layers.Layer):
    """
  Straight-forward, fully connected set of neural nets to map some input to outputs.

  Can act as encoder or decoder depending on settings.
  DOES account for periodic degrees of freedom, converting to sine-cosine pairs if any periodic dofs indicated.
  Note that this could apply to either periodic inputs or latent degrees of freedom
  (if, say, prior based on von Mises).
  Note that this just outputs some number of parameters, which may or may not correspond to the latent dimensionality
  for an encoder or data dimensionality for a decoder.
  The actually dimensionality will depend on how the encoder or decoder distribution operates on the outputs of
  this layer.

  Attributes
  ----------
  target_shape : tuple
      The output shape of this layer.
  hidden_dim : list
      A list of the hidden dimensions of hidden neural network layers.
  periodic_dofs : tf.Tensor
      A boolean tensor that indicates which degrees of freedom are periodic.
  batch_norm : bool
      Whether batch normalization layers are present.
  """

    def __init__(self,
                 target_shape,
                 hidden_dim=200,
                 periodic_dofs=False,
                 batch_norm=False,
                 name='mapping',
                 activation=tf.nn.relu,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        """
    Creates a FCDeepNN layer instance

    Parameters
    ----------
    target_shape : int or tuple
        Number of outputs, intended as a flat output of shape (batch_shape, target_shape),
        or, if provided with a tuple or list, that shape appended to batch_shape (-1,)+target_shape
    hidden_dim : int or list of ints, default 1200
        Dimension of hidden layer(s). Will create hidden layer for each value if it is a list.
    periodic_dofs : bool or list of bool, default False
        A mask with True appearing where have periodic dofs. By default, this is False for all dofs,
        and if passed just True, will treat all as periodic.
    batch_norm : bool, default False
        Whether or not to apply batch normalization layers in between dense layers.
    """
        super(FCDeepNN, self).__init__(name=name, **kwargs)

        try:
            self.target_shape = tuple(target_shape)
        except TypeError:
            self.target_shape = (target_shape, )

        # Need to define hidden dimensions and create layer for each
        if isinstance(hidden_dim, int):
            self.hidden_dim = [
                hidden_dim,
            ]
        else:
            self.hidden_dim = hidden_dim

        self.periodic_dofs = periodic_dofs
        self.batch_norm = batch_norm

        # Specify activation (on hidden layers) and kernel initializer
        self.activation = activation
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        # Set up periodic DOF boolean list
        if isinstance(self.periodic_dofs, bool):
            self.any_periodic = self.periodic_dofs
            self.periodic_dofs = tf.convert_to_tensor([
                self.periodic_dofs,
            ] * np.prod(input_shape[1:]))
        else:
            # Check that shape matches input
            if len(self.periodic_dofs) != np.prod(input_shape[1:]):
                raise ValueError("Shape of periodic_dofs (%i) should match flattened input (%i)." %
                                 (len(self.periodic_dofs), np.prod(input_shape[1:])))
            self.any_periodic = np.any(self.periodic_dofs)
            self.periodic_dofs = tf.convert_to_tensor(self.periodic_dofs)

        # Define all dense layers (input shapes will be handled during build automatically)
        self.flattened = tf.keras.layers.Flatten()

        # Create layer for each hidden dimension
        self.layer_list = []
        for hd in self.hidden_dim:
            self.layer_list.append(
                tf.keras.layers.Dense(hd, activation=self.activation, kernel_initializer=self.kernel_initializer))
            if self.batch_norm:
                self.layer_list.append(tf.keras.layers.BatchNormalization())

        # Add a layer without activation at the end
        self.layer_list.append(
            tf.keras.layers.Dense(np.prod(self.target_shape),
                                  activation=None,
                                  kernel_initializer=self.kernel_initializer))

        # And add a reshaping layer
        self.layer_list.append(tf.keras.layers.Reshape(self.target_shape))

    def call(self, inputs, training=False):
        """
    Applies neural network mapping (or transformation) to inputs.

    Parameters
    ----------
    inputs : tf.Tensor
        Inputs to this layer
    training : bool, default False
        Whether or not training or generating. Applicable if have batch normalization layers.

    Returns
    -------
    out : tf.Tensor
        The output of applying this transformation.
    """
        out = self.flattened(inputs)

        # If have periodic DOFs, want to convert to 2D non-periodic coordinates in first step
        if self.any_periodic:
            out_p = tf.boolean_mask(out, self.periodic_dofs, axis=1)
            out_nonp = tf.boolean_mask(out, tf.math.logical_not(self.periodic_dofs), axis=1)
            cos_p = tf.math.cos(out_p)
            sin_p = tf.math.sin(out_p)
            out = tf.concat([out_nonp, cos_p, sin_p], axis=-1)

        # Now apply all layers
        for layer in self.layer_list:
            out = layer(out, training=training)

        return out

    def get_config(self):
        config = super(FCDeepNN, self).get_config()
        config.update({
            "target_shape": self.target_shape,
            "hidden_dim": self.hidden_dim,
            "periodic_dofs": self.peridic_dofs,
            "batch_norm": self.batch_norm,
        })
        return config


# May be helpful, but mostly intended as a template
class CGCentroid(tf.keras.layers.Layer):
    """
  Generically defines a CG mapping from residues (or molecules) to their centroids.

  Will be defined for the specific system at hand based on a list of the numbers of
  atoms in each residue/molecule.

  Attributes
  ----------
  res_atom_nums : list-like of ints
      Number of atoms in each residue or molecule.
  """

    def __init__(self, res_atom_nums, name='cg_centroid', **kwargs):
        """
    Creates a CGCentroid layer instance

    Parameters
    ----------
    res_atom_nums : list-like containing ints
        List defining the number of atoms in each residue/molecule. This should be in the
        correct order of molecules for provided coordinate inputs, so that if there are
        N atoms in the first residue/molecule, the first N coordinates should relate to that molecule.
        Can be a list or array or other subscriptable object containing ints.
    """

        super(CGCentroid, self).__init__(name=name, **kwargs)

        self.res_atom_nums = res_atom_nums

    def call(self, inputs):
        """
    Maps fine-grained (atomistic) coordinates to CG coordinates based on their centroid.

    Parameters
    ----------
    inputs : tf.Tensor
        Coordinates of all residues/molecules in rectangular coordinates (shape should be batch x N_atoms x 3).

    Returns
    -------
    out : tf.Tensor
        Means (centroids) of atom positions in each residue (shape is batch x N_res x 3).
    """
        res_atoms = tf.split(inputs, self.res_atom_nums,
                             axis=-2)  # returns a tf-wrapped list we can operate on with tf functions
        centroids = []
        # Loop, but really need to figure out how to apply as mapping or with ragged tensors
        for ra in res_atoms:
            centroids.append(tf.reduce_mean(ra, axis=-2))
        centroids = tf.stack(centroids, axis=0)
        return tf.transpose(centroids, perm=[1, 0, 2])

    def get_config(self):
        config = super(CGCentroid, self).get_config()
        config.update({
            "res_atom_nums": self.res_atom_nums,
        })
        return config


# Possibly more helpful, but still can't convert for something like the Rosetta centroid representation
# (that preserves all backbone atoms, but then uses the centroid for all atoms past CB,
# so not sure about alanine or glycine)
class CGCenterOfMass(tf.keras.layers.Layer):
    """
  Generically defines a CG mapping from residues (or molecules) to their centers of mass.

  This actually requires that the residue list is supplied as part of the input to this layer.
  This way, we look up numbers of atoms and masses/weights in dictionaries, then apply the transformation.

  Attributes
  ----------
  res_atom_nums : dict
      A dictionary mapping residue (or molecule) names to numbers of atoms.
  res_masses : dict
      A dictionary mapping residue names to a list-like of the masses of each atom.
  """

    def __init__(self, res_atom_nums, res_masses=None, name='cg_com', **kwargs):
        """
    Creates a CGCenterOfMass layer instance

    Parameters
    ----------
    res_atom_nums : dict
        Dictionary mapping residue (or molecule) names to numbers of atoms.
    res_masses : dict
        Dictionary mapping residue names to a list-like of the masses of each atom.
    """

        super(CGCenterOfMass, self).__init__(name=name, **kwargs)

        self.res_atom_nums = res_atom_nums
        if res_masses is None:
            self.res_masses = dict((k, np.ones(v)) for (k, v) in res_atom_nums.items())
        else:
            self.res_masses = res_masses
        self.res_weights = dict((k, tf.reshape(tf.convert_to_tensor(v / np.sum(v), dtype='float32'), (1, -1, 1)))
                                for (k, v) in self.res_masses.items())

    def call(self, coords, res_names):
        """
    Maps fine-grained (atomistic) coordinates to CG beads based on the atoms' center of mass.

    Parameters
    ----------
    coords : tf.Tensor
        Coordinates of all residues/molecules in rectangular coordinates (shape should be batch x N_atoms x 3).
    res_names : list-like
        List-like of residue/molecule names in an order matching the coordinates. Can be any subscriptable that
        returns the residue names involved in this input. It is assumed that the same list of residues is
        applied to every member of the batch.

    Returns
    -------
    out : tf.Tensor
        Center of mass of each residue (shape is batch x N_res x 3).
    """
        # Looping and putting together at end
        # Really need to figure out ragged tensors, I think, or nested mapping operators in tensorflow
        atom_nums = [self.res_atom_nums[r] for r in res_names]
        res_atoms = tf.split(coords, atom_nums,
                             axis=-2)  # returns a tf-wrapped list we can operate on with tf functions
        coms = []
        for i, r in enumerate(res_names):
            coms.append(tf.reduce_sum(res_atoms[i] * self.res_weights[r], axis=-2))
        coms = tf.stack(coms, axis=0)
        return tf.transpose(coms, perm=[1, 0, 2])

    def get_config(self):
        config = super(CGCentroid, self).get_config()
        config.update({
            "res_atom_nums": self.res_atom_nums,
            "res_masses": self.res_masses,
        })
        return config


class DistanceSelection(tf.keras.layers.Layer):
    """
  Layer to apply a distance-based mask to coordinates.

  Only coordinates within the specified cutoff will be kept, with padding applied to make
  the output have max_included particles. In other words, the input should be of shape
  (N_batch, N_particles, 3) and the output will be of shape (N_batch, max_included, 3).
  To reach max_included, zeros may be added to pad the particle coordinates within the
  cutoff. If more than max_included particles are in the cutoff, they will be truncated,
  sorted so that the nearest particles are ensured to be included. Having a fixed size is
  necessary for further operations with neural networks. So if you want this function to
  serve as only a distance-based cutoff, mmax_included should be selected to ensure that
  it is always as large or larger than the maximum number of particles within the specified
  cutoff. The reference coordinates must also be supplied when calling this layer, and
  should be in the shape (N_batch, 3), or at least reshapable to (N_batch, 1, 3).

  Attributes
  ----------
  cutoff : float
      The maximum distance of particles from the reference to consider.
  max_included : int
      Specifies the maximum number of particles within the cutoff to consider.
  box_lengths : tf.Tensor
      A length-3 tensor specifying the simulation box dimensions.
  """

    def __init__(self, cutoff, max_included=50, box_lengths=None, name='dist_select', **kwargs):
        """
    Creates DistanceSelection layer instance.

    Parameters
    ----------
    cutoff : float
        The distance cutoff with particles closer than this distance included.
    max_included : int, default 50
        Maximum number of particles within the cutoff. Can determine from physics (RDF, etc.).
        If want to function as only distance-based cutoff, set to more than expected number.
        Otherwise, applies both number AND distance-based cutoff.
    box_lengths : list-like, default None
        A list, array, or tensor with 3 elements representing the simulation box edge lengths.
        If provided, assumes the box is a fixed size and wraps periodically when computing
        distances.
    """
        super(DistanceSelection, self).__init__(name=name, **kwargs)

        self.cutoff = cutoff
        self.sq_cut = cutoff**2
        self.max_included = max_included
        if box_lengths is not None:
            self.box_lengths = tf.constant(box_lengths)
            self.box_lengths = tf.reshape(self.box_lengths, (1, 1, 3))
        else:
            self.box_lengths = None

    def call(self, coords, ref, box_lengths=None, particle_info=None):
        """
    Selects all particles within the cutoff distance and pads (or truncates).

    If particle_info is also provided, also masks and pads/truncates that information.

    Parameters
    ----------
    coords : tf.Tensor or tf.RaggedTensor
        Coordinates to select from based on the distance cutoff (N_batch x N_particles x 3).
        Note that this could also be a nested list or ragged tensor where the number
        of particles is different for each batch instance.
    ref : tf.Tensor
        Reference coordinates for each batch (N_batch x 3).
    box_lengths : tf.Tensor, default None
        Uses these box lengths calculating distances with periodic cell. This is necessary to
        provide if the box changes size for each configuration, such as for the NPT ensemble.
        Should be of shape (N_batch, 3) if provided.
    particle_info : tf.Tensor, default None
        Extra particle information that will be masked in the same way as the
        coordinates. Could be things like particle type, LJ parameters, charge, etc.

    Returns
    -------
    select_coords : tf.Tensor
        Selected coordinates within the distance cutoff, padded with zeros to have
        shape (N_batch, max_included, 3).
    """
        # Want to work with ragged tensor so can apply cutoff to each set of coordinates
        # (which may be point clouds of different sizes for different batch elements)
        if not isinstance(coords, tf.RaggedTensor):
            coords = tf.RaggedTensor.from_tensor(coords)

        if particle_info is not None:
            if not isinstance(particle_info, tf.RaggedTensor):
                particle_info = tf.RaggedTensor.from_tensor(particle_info)

        batch_size = tf.shape(coords)[0]

        # Reshape references (also serves as check of shape, but lazily don't catch and provide specific error)
        ref = tf.reshape(ref, (batch_size, 1, 3))

        local_coords = coords - ref

        # Apply periodic wrapping if have box lengths
        # Try those provided to call first, then use stored (if have either)
        if box_lengths is not None:
            box_lengths = tf.reshape(box_lengths, (batch_size, 1, 3))
            local_coords = local_coords - box_lengths * tf.round(local_coords / box_lengths)
        elif self.box_lengths is not None:
            local_coords = local_coords - self.box_lengths * tf.round(local_coords / self.box_lengths)

        # Since applying self.max_included may remove some within cutoff, make sure priortize nearest
        # To do that, need to sort, but can only sort regular tensors, not ragged
        # Will pad to largest size necessary (config with most particles) with large numbers
        local_coords = local_coords.to_tensor(default_value=tf.float32.max)

        # And if max particles is less than self.max_included, must pad
        max_shape = tf.shape(local_coords)[1]
        if max_shape < self.max_included:
            local_coords = tf.pad(
                local_coords,
                [[0, 0], [0, self.max_included - max_shape], [0, 0]],
                constant_values=tf.float32.max,
            )

        # Get squared distances
        dists_sq = tf.reduce_sum(local_coords * local_coords, axis=-1)

        # Take nearest self.max_included based on indices
        # Negate since top_k takes largest values
        near_dists, near_inds = tf.math.top_k(-dists_sq, k=self.max_included)

        # Select out nearest coordinates
        select_coords = tf.gather(local_coords, near_inds, axis=1, batch_dims=1)

        # Mask based on cutoff
        # Expand dimensions of mask so works with tf.where
        mask = tf.expand_dims((-near_dists <= self.sq_cut), axis=-1)
        select_coords = tf.where(mask, select_coords, tf.zeros_like(select_coords))

        if particle_info is not None:
            particle_info = particle_info.to_tensor(default_value=0.0)
            if max_shape < self.max_included:
                particle_info = tf.pad(
                    particle_info,
                    [[0, 0], [0, self.max_included - max_shape], [0, 0]],
                    constant_values=0.0,
                )
            select_info = tf.gather(particle_info, near_inds, axis=1, batch_dims=1)
            select_info = tf.where(mask, select_info, tf.zeros_like(select_info))
            return select_coords, select_info
        else:
            return select_coords

    def get_config(self):
        config = super(DistanceSelection, self).get_config()
        config.update({
            "cutoff": self.cutoff,
            "max_included": self.max_included,
            "box_lengths": self.box_lengths,
        })
        return config


# If no FG coordinates for some batch configurations, will fail currently
# Need to find way to pass in ragged tensor with some dimensions of zero
# Can do!
# Need to get list of number of particles in each FG (or CG configuration)
# Then np.vstack the particle configurations into a single array of sum(n_particles) x 3
# Compute the cumulative sum over the number of particles, making sure the resulting list starts
# with zero
# Then apply tf.RaggedTensor.from_row_splits(stacked_coords, row_splits=cumulative_sum)
# It works because arrays with no particles will appear as (0, 3) in their shape, so will add 0 to the sum
# and then will be input as (0, 3) tensors in the ragged tensor as well.
# It turns out that RaggedTensor handles empty (zero-length) entries by just ignoring them for subtraction, etc.


class AttentionBlock(tf.keras.layers.Layer):
    """
  Geometric algebra attention block as described in Spellings (2021).

  The operations on point clouds are rotationally invariant and permutationally equivariant.

  Attributes
  ----------
  hidden_dim : int
      The dimension of hidden neural network layers.
  """

    def __init__(self, hidden_dim=40, name='geom_attn', activation=tf.nn.relu, **kwargs):
        """
    Creates an AttentionBlock layer instance

    Parameters
    ----------
    hidden_dim : int, default 40
        Hidden dimension of dense network after attention.
    """
        super(AttentionBlock, self).__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.activation = activation
        self.supports_masking = True

    def build(self, input_shape):
        working_dim = input_shape[1][-1]
        self.score_fun = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(self.hidden_dim, activation=self.activation),
             tf.keras.layers.Dense(1)])
        self.value_fun = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation(self.activation),
            tf.keras.layers.Dense(working_dim)
        ])
        self.attn = gaa_keras.VectorAttention(
            self.score_fun,
            self.value_fun,
            reduce=False,
            merge_fun='concat',
            join_fun='concat',
            rank=2,
        )
        self.nonlinearity = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation(self.activation),
            tf.keras.layers.Dense(working_dim)
        ])

    def call(self, inputs, mask=None):
        """
    Performs geometric algebra attention to a point cloud and current embedding to create a new embedding.

    Parameters
    ----------
    inputs : tf.Tensor
        List of coords (coordinates of particles) and embedding (current embedding or particle information).

    Returns
    -------
    new_embed : tf.Tensor
        The new embedding after applying this attention block.
    """
        # Inputs as list to get input_shape for both as list
        # Otherwise just get input shape of first argument
        coords = inputs[0]
        embedding = inputs[1]
        new_embed = self.attn([coords, embedding], mask=mask)
        new_embed = self.nonlinearity(new_embed)
        new_embed = new_embed + embedding
        return new_embed

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
        })
        return config


class ParticleEmbedding(tf.keras.layers.Layer):
    """
  An embedding of CG or FG particles from their Cartesian coordinates to a new space.

  Geometric algebra attention (https://github.com/klarh/geometric_algebra_attention) is used to ensure
  that the embedding is permutation equivariant and rotation invariant.
  Note that translation invariance should also be applied by only passing local coordinates (relative to
  some reference site) to this function.

  Attributes
  ----------
  embedding_dim : int
      Dimension of ultimate embedding.
  hidden_dim : int
      Dimension of hidden neural network layers.
  num_blocks : int
      Number of geometric attention blocks applied.
  mask_zero : bool
      Whether or not a mask will be applied to zeros.
  """

    def __init__(self,
                 embedding_dim,
                 hidden_dim=40,
                 num_blocks=2,
                 mask_zero=True,
                 name='particle_embedding',
                 activation=tf.nn.relu,
                 **kwargs):
        """
    Creates ParticleEmbedding layer instance

    Parameters
    ----------
    embedding_dim : int
        Final dimension of embedding; also used for working dimension of extra information.
    hidden_dim : int, default 40
        Dimension of all hidden dimensions in dense networks.
    num_blocks : int, default 2
        Number of attention blocks we will apply.
    mask_zero : bool, default True
        Whether or not to apply a mask layer to mask out zeros.
    """
        super(ParticleEmbedding, self).__init__(name=name, **kwargs)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.mask_zero = mask_zero
        self.activation = activation

    def build(self, input_shape):

        # Network for mapping input extra info (particle identities,parameters, etc.) to working dimensionality
        self.info_net = tf.keras.layers.Dense(self.embedding_dim)  # No activation, just linear mapping

        # Build attention blocks
        self.block_list = []
        for i in range(self.num_blocks):
            self.block_list.append(AttentionBlock(self.hidden_dim, activation=self.activation))

        if self.mask_zero:
            self.mask = tf.keras.layers.Masking()
        else:
            self.mask = None

        # Add one more attention layer with permutation invariance to sum over particles for embeddings
        # Consider a wrapper for just creating attention, since always need sequential models like this
        # Could then also use that in AttentionBlock, though there reduce=False, not True like here
        self.final_attn = gaa_keras.VectorAttention(
            tf.keras.models.Sequential(
                [tf.keras.layers.Dense(self.hidden_dim, activation=self.activation),
                 tf.keras.layers.Dense(1)]),
            tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.hidden_dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.Dense(self.embedding_dim)
            ]),
            reduce=True,
            merge_fun='concat',
            join_fun='concat',
            rank=2,
        )

    def call(self, coords, particle_info):
        """
    Creates an embedding based on rotation invariant and permutation invariant operations.

    Parameters
    ----------
    coords : tf.Tensor
        Coordinates of particles.
    particle_info : tf.Tensor
        Particle information like type or parameters.

    Returns
    -------
    embedding : tf.Tensor
        The embedding of these particles.
    """
        if self.mask_zero:
            # Can only mask coordinates properly
            # Masking both paddings and any coordinates at origin (just CG bead we're backmapping, so ignore)
            coords = self.mask(coords)

        embedding = self.info_net(
            particle_info)  # Won't be masked, but that's ok because gets masked in attention layer

        for block in self.block_list:
            embedding = block([coords, embedding])

        embedding = self.final_attn([coords, embedding])

        return embedding

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "mask_zero": self.mask_zero,
        })
        return config


class LocalParticleDescriptors(tf.keras.layers.Layer):
    """
  A layer defining an embedding describing the local environment of a point cloud around a reference.

  This class joins together distance masking and embedding layers to take Cartesian particle
  coordinates, mask them by distance to a reference while converting to local coordinates, and then
  compute embeddings, or descriptors. Likely, the mask and embed functions will be DistanceSelection
  and ParticleEmbedding objects. Mainly a wrapper for convenience since will use these functions
  together quite a bit.

  Attributes
  ----------
  mask_fn : callable (layer)
      A layer applying a distnace-based masking to define those particles local around a reference.
  embed_fn : callable (layer)
      A layer that turns the local point cloud into an embedding.
  """

    def __init__(self, mask_fn, embed_fn, name='local_particle_desc', **kwargs):
        """
    Creates a LocalParticleDescriptors layer instance

    Parameters
    ----------
    mask_fn : callable
        The distance-based masking function that also produces local coordinates of selected particles.
        This should likely be a layer, or at least a tf.Module for trainability in tensorflow.
    embed_fn : callable
        The embedding function to convert local coordinates to descriptors. Again, a keras layer will
        work the best.
    """
        super(LocalParticleDescriptors, self).__init__(name=name, **kwargs)

        self.mask_fn = mask_fn
        self.embed_fn = embed_fn

    def call(self, coords, ref, props, box_lengths=None):
        """
    Applies a masking around the reference and creates a local embedding of that local point cloud.

    Parameters
    ----------
    coords : tf.Tensor or tf.RaggedTensor
        Coordinates of particles to mask by distance (N_batch, (N_particles), 3).
        Note that this can be a ragged tensor where N_particles is different for each batch index.
    ref : tf.Tensor
        Coordinates of reference positions for determining mask (N_batch, 1, 3).
        Only one reference coordinate for each batch element.
    props : tf.Tensor or tf.RaggedTensor
        Properties of particles, such as type, or parameters like charge (N_batch, (N_particles), N_props).
        Can be ragged like coords, but first two dimensions, batch and number of particles, should match coords.
    box_lengths : tf.Tensor, default None
        Optional argument for box dimensions if box changes with given configuration and want to
        wrap periodically when computing distances

    Returns
    -------
    descriptors : tf.Tensor
        Descriptors of the local environment around each reference position (N_batch, N_descriptors).
        Note that the number of descriptors is given by the output dimension of self.embed_fn.
    """
        local_coords, local_props = self.mask_fn(coords, ref, particle_info=props, box_lengths=box_lengths)
        descriptors = self.embed_fn(local_coords, local_props)
        return descriptors

    def get_config(self):
        config = super(LocalParticleDescriptors, self).get_config()
        config.update({
            "mask_fn": self.mask_fn,
            "embed_fn": self.embed_fn,
        })
        return config
