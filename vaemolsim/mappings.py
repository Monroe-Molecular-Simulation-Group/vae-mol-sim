
import numpy as np
import tensorflow as tf

from geometric_algebra_attention import keras as gaa_keras


#Transformations via neural networks, etc.
#Can have some for encoders, some for decoders
#Will mostly be based on sequential keras layers
#But could also include selecting k-nearest neighbors
#Or neighbors within a cutoff
#These could be part of the transformation/mapping

class FCDeepNN(tf.keras.layers.Layer):
  """
  Straight-forward, fully connected set of neural nets to map some input to outputs.
  Can act as encoder or decoder depending on settings.
  DOES account for periodic degrees of freedom, converting to sine-cosine pairs if any periodic dofs indicated.
  Note that this could apply to either periodic inputs or latent degrees of freedom (if, say, prior based on von Mises).
  Note that this just outputs some number of parameters, which may or may not correspond to the latent dimensionality
  for an encoder or data dimensionality for a decoder.
  The actually dimensionality will depend on how the encoder or decoder distribution operates on the outputs of this layer.
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
    Inputs:
        target_shape - (int or tuple) number of outputs, intended as a flat output of shape (batch_shape, target_shape),
                       or, if provided with a tuple or list, that shape appended to batch_shape (-1,)+target_shape
        hidden_dim - (int or list of ints, 1200) dimension of hidden layer(s); will create hidden layer for each
                     value if it is a list
        periodic_dofs - (bool or list of bool, False) a mask with True appearing where have periodic dofs; by
                        default, this is False for all dofs, and if passed just True, will treat all as periodic
        batch_norm - (bool, False) whether or not to apply batch normalization layers in between dense layers
    Outputs:
        FCDeepNN layer
    """
    super(FCDeepNN, self).__init__(name=name, **kwargs)

    try:
      self.target_shape = tuple(target_shape)
    except TypeError:
      self.target_shape = (target_shape,)

    #Need to define hidden dimensions and create layer for each
    if isinstance(hidden_dim, int):
      self.hidden_dim = [hidden_dim,]
    else:
      self.hidden_dim = hidden_dim

    self.periodic_dofs = periodic_dofs
    self.batch_norm = batch_norm

    #Specify activation (on hidden layers) and kernel initializer
    self.activation = activation
    self.kernel_initializer = kernel_initializer

  def build(self, input_shape):
    #Set up periodic DOF boolean list
    if isinstance(self.periodic_dofs, bool):
      self.any_periodic = self.periodic_dofs
      self.periodic_dofs = tf.convert_to_tensor([self.periodic_dofs,]*np.prod(input_shape[1:]))
    else:
      #Check that shape matches input
      if len(self.periodic_dofs) != np.prod(input_shape[1:]):
        raise ValueError("Shape of periodic_dofs (%i) should match flattened input (%i)."%(len(self.periodic_dofs), np.prod(input_shape[1:])))
      self.any_periodic = np.any(self.periodic_dofs)
      self.periodic_dofs = tf.convert_to_tensor(self.periodic_dofs)

    #Define all dense layers (input shapes will be handled during build automatically)
    self.flattened = tf.keras.layers.Flatten()

    #Create layer for each hidden dimension
    self.layer_list = []
    for hd in self.hidden_dim:
      self.layer_list.append(tf.keras.layers.Dense(hd, activation=self.activation,
                                    kernel_initializer=self.kernel_initializer))
      if self.batch_norm:
        self.layer_list.append(tf.keras.layers.BatchNormalization())

    #Add a layer without activation at the end
    self.layer_list.append(tf.keras.layers.Dense(np.prod(self.target_shape), activation=None,
                                       kernel_initializer=self.kernel_initializer))

    #And add a reshaping layer
    self.layer_list.append(tf.keras.layers.Reshape(self.target_shape))


  def call(self, inputs, training=False):
    out = self.flattened(inputs)

    #If have periodic DOFs, want to convert to 2D non-periodic coordinates in first step
    if self.any_periodic:
      out_p = tf.boolean_mask(out, self.periodic_dofs, axis=1)
      out_nonp = tf.boolean_mask(out, tf.math.logical_not(self.periodic_dofs), axis=1)
      cos_p = tf.math.cos(out_p)
      sin_p = tf.math.sin(out_p)
      out = tf.concat([out_nonp, cos_p, sin_p], axis=-1)

    #Now apply all layers
    for layer in self.layer_list:
      out = layer(out, training=training)

    return out

  def get_config(self):
    config = super(FCDeepNN, self).get_config()
    config.update({"target_shape": self.target_shape,
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
  """

  def __init__(self,
               res_atom_nums,
               name='cg_centroid',
               **kwargs):
    """
    Inputs:
        res_atom_nums - (list-like) list defining the number of atoms in each residue/molecule;
                        this should be in the correct order of molecules for provided coordinate inputs,
                        so that if there are N atoms in the first residue/molecule, the first N
                        coordinates should relate to that molecule
    Outputs:
        CGCentroid object
    """

    super(CGCentroid, self).__init__(name=name, **kwargs)

    self.res_atom_nums = res_atom_nums
  
  def call(self, inputs):
    """
    Inputs:
        inputs - coordinates of all residues/molecules in rectangular coordinates (shape should be batch x N_atoms x 3)
    Outputs:
        out - means (centroid) of atom positions in each residue (shape is batch x N_res x 3)
    """
    res_atoms = tf.split(inputs, self.res_atom_nums, axis=-2) #returns a tf-wrapped list we can operate on with tf functions
    centroids = []
    # Loop, but really need to figure out how to apply as mapping or with ragged tensors
    for ra in res_atoms:
      centroids.append(tf.reduce_mean(ra, axis=-2))
    centroids = tf.stack(centroids, axis=0)
    return tf.transpose(centroids, perm=[1, 0, 2])

  def get_config(self):
    config = super(CGCentroid, self).get_config()
    config.update({"res_atom_nums": self.res_atom_nums,
                  })
    return config


# Possibly more helpful, but still can't convert for something like the Rosetta centroid representation
# (that preserves all backbone atoms, but then uses the centroid for all atoms past CB, so not sure about alanine or glycine)
class CGCenterOfMass(tf.keras.layers.Layer):
  """
  Generically defines a CG mapping from residues (or molecules) to their centers of mass.
  This actually requires that the residue list is supplied as part of the input to this layer.
  This way, we look up numbers of atoms and masses/weights in dictionaries, then apply the transformation.
  """

  def __init__(self,
               res_atom_nums,
               res_masses=None,
               name='cg_com',
               **kwargs):
    """
    Inputs:
        res_atom_nums - (dict) dictionary mapping residue (or molecule) names to numbers of atoms
        res_masses - (dict) dictionary mapping residue names to a list-like of the masses of each atom
    Outputs:
        CGCenterOfMass object
    """

    super(CGCenterOfMass, self).__init__(name=name, **kwargs)

    self.res_atom_nums = res_atom_nums
    if res_masses is None:
      self.res_masses = dict((k, np.ones(v)) for (k, v) in res_atom_nums.items())
    else:
      self.res_masses = res_masses
    self.res_weights = dict((k, tf.reshape(tf.convert_to_tensor(v/np.sum(v), dtype='float32'), (1, -1, 1))) for (k, v) in self.res_masses.items())
  
  def call(self, coords, res_names):
    """
    Inputs:
        coords - coordinates of all residues/molecules in rectangular coordinates (shape should be batch x N_atoms x 3)
        res_names - list-like of residue/molecule names in an order matching the coordinates
    Outputs:
        out - center of mass of each residue (shape is batch x N_res x 3)
    """
    # Looping and putting together at end
    # Really need to figure out ragged tensors, I think, or nested mapping operators in tensorflow
    atom_nums = [self.res_atom_nums[r] for r in res_names]
    res_atoms = tf.split(coords, atom_nums, axis=-2) #returns a tf-wrapped list we can operate on with tf functions
    coms = []
    for i, r in enumerate(res_names):
      coms.append(tf.reduce_sum(res_atoms[i]*self.res_weights[r], axis=-2))
    coms = tf.stack(coms, axis=0)
    return tf.transpose(coms, perm=[1, 0, 2])

  def get_config(self):
    config = super(CGCentroid, self).get_config()
    config.update({"res_atom_nums": self.res_atom_nums,
                   "res_masses": self.res_masses,
                  })
    return config


class DistanceSelection(tf.keras.layers.Layer):
  """
  Layer to apply a distance-based mask to coordinates. Only coordinates within the specified
  cutoff will be kept, with padding applied to make the output have max_included particles.
  In other words, the input should be of shape (N_batch, N_particles, 3) and the output
  will be of shape (N_batch, max_included, 3). To reach max_included, zeros may be added
  to pad the particle coordinates within the cutoff. Having a fixed size is necessary for
  further operations with neural networks, so max_included should be selected to ensure
  that it is always as large or larger than the maximum number of particles within the
  specified cutoff. The reference coordinates must also be supplied when calling this
  layer, and should be in the shape (N_batch, 3), or at least reshapable to
  (N_batch, 1, 3).
  """

  def __init__(self,
               cutoff,
               max_included=50,
               box_lengths=None,
               name='dist_select',
               **kwargs):
    """
    Inputs:
        cutoff - the distance cutoff with particles closer than this distance included
        max_included - (50) maximum number of particles within the cutoff; can determine from physics (RDF, etc.)
        box_lengths - (None) simulation box edge lengths; wraps periodically if provided
    Outputs:
        DistanceSelection object
    """
    super(DistanceSelection, self).__init__(name=name, **kwargs)
    
    self.cutoff = cutoff
    self.sq_cut = cutoff**2
    self.max_included = max_included
    self.box_lengths = box_lengths
    if self.box_lengths is not None:
      self.box_lengths = tf.reshape(self.box_lengths, (1, 1, 3))

  def call(self, coords, ref, box_lengths=None, particle_info=None):
    """
    Inputs:
        coords - coordinates to select from based on the distance cutoff (N_batch x N_particles x 3);
                 note that this could also be a nested list or ragged tensor where the number
                 of particles is different for each batch instance
        ref - reference coordinates for each batch (N_batch x 3)
        box_lengths - (None) uses these box lengths calculating distances with periodic cell;
                      should be of shape (N_batch, 3) if provided
        particle_info - (None) extra particle information that will be masked in the same way as the
                        coordinates; could be things like particle type, LJ parameters, charge, etc.
    Outputs:
        select_coords - selected coordinates within the distance cutoff, padded with zeros to have
                        shape (N_batch, max_included, 3)
    """
    #Want to work with ragged tensor so can apply cutoff to each set of coordinates
    #(which may be point clouds of different sizes for different batch elements)
    if not isinstance(coords, tf.RaggedTensor):
      coords = tf.RaggedTensor.from_tensor(coords)

    if particle_info is not None:
      if not isinstance(particle_info, tf.RaggedTensor):
        particle_info = tf.RaggedTensor.from_tensor(particle_info)
    
    batch_size = tf.shape(coords)[0]

    #Reshape references (also serves as check of shape, but lazily don't catch and provide specific error)
    ref = tf.reshape(ref, (batch_size, 1, 3))

    local_coords = coords - ref

    #Apply periodic wrapping if have box lengths
    #Try those provided to call first, then use stored (if have either)
    if box_lengths is not None:
      box_lengths = tf.reshape(box_lengths, (batch_size, 1, 3))
      local_coords = local_coords - box_lengths*tf.round(local_coords / box_lengths)
    elif self.box_lengths is not None:
      local_coords = local_coords - self.box_lengths*tf.round(local_coords / self.box_lengths)

    #Get squared distances
    dists_sq = tf.reduce_sum(local_coords * local_coords, axis=-1)

    #Mask based on cutoff
    #Pad coords as needed, returning tensor, not ragged tensor
    mask = (dists_sq <= self.sq_cut)
    select_coords = tf.ragged.boolean_mask(local_coords, mask).to_tensor(default_value=0.0, shape=(batch_size, self.max_included, 3))

    if particle_info is not None:
      select_info = tf.ragged.boolean_mask(particle_info, mask).to_tensor(default_value=0.0,
                                                              shape=(batch_size, self.max_included, particle_info.shape[-1]))
      return select_coords, select_info
    else:
      return select_coords

  def get_config(self):
    config = super(DistanceSelection, self).get_config()
    config.update({"cutoff": self.cutoff,
                   "max_included": self.max_included,
                   "box_lengths": self.box_lengths,
                  })
    return config

#If no FG coordinates for some batch configurations, will fail currently
#Need to find way to pass in ragged tensor with some dimensions of zero
#Can do!
#Need to get list of number of particles in each FG (or CG configuration)
#Then np.vstack the particle configurations into a single array of sum(n_particles) x 3
#Compute the cumulative sum over the number of particles, making sure the resulting list starts
#with zero
#Then apply tf.RaggedTensor.from_row_splits(stacked_coords, row_splits=cumulative_sum)
#It works because arrays with no particles will appear as (0, 3) in their shape, so will add 0 to the sum
#and then will be input as (0, 3) tensors in the ragged tensor as well.
#It turns out that RaggedTensor handles empty (zero-length) entries by just ignoring them for subtraction, etc.


class AttentionBlock(tf.keras.layers.Layer):
  """
  Geometric algebra attention block as described in Spellings (2021)
  """

  def __init__(self,
               hidden_dim=40,
               name='geom_attn',
               activation=tf.nn.relu,
               **kwargs):
    """
    Inputs:
        hidden_dim - hidden dimension of dense network after attention
    Outputs:
        AttentionBlock object
    """
    super(AttentionBlock, self).__init__(name=name, **kwargs)

    self.hidden_dim = hidden_dim
    self.activation = activation
    self.supports_masking = True

  def build(self, input_shape):
    working_dim = input_shape[1][-1]
    self.score_fun = tf.keras.models.Sequential([tf.keras.layers.Dense(self.hidden_dim, activation=self.activation),
                                                 tf.keras.layers.Dense(1)
                                                ])
    self.value_fun = tf.keras.models.Sequential([tf.keras.layers.Dense(self.hidden_dim),
                                                 tf.keras.layers.LayerNormalization(),
                                                 tf.keras.layers.Activation(self.activation),
                                                 tf.keras.layers.Dense(working_dim)
                                                ])
    self.attn = gaa_keras.VectorAttention(self.score_fun, self.value_fun,
                                              reduce=False,
                                              merge_fun='concat',
                                              join_fun='concat',
                                              rank=3,
                                             )
    self.nonlinearity = tf.keras.models.Sequential([tf.keras.layers.Dense(self.hidden_dim),
                                                 tf.keras.layers.LayerNormalization(),
                                                 tf.keras.layers.Activation(self.activation),
                                                 tf.keras.layers.Dense(working_dim)
                                                ])
  
  def call(self, inputs, mask=None):
    """
    Inputs:
        list of coords (coordinates of particles) and embedding (current embedding or particle information)
    Outputs:
        new_embed - the new embedding after applying this attention block
    """
    #Inputs as list to get input_shape for both as list
    #Otherwise just get input shape of first argument
    coords = inputs[0]
    embedding = inputs[1]
    new_embed = self.attn([coords, embedding], mask=mask)
    new_embed = self.nonlinearity(new_embed)
    new_embed = new_embed + embedding
    return new_embed

  def get_config(self):
    config = super(AttentionBlock, self).get_config()
    config.update({"hidden_dim": self.hidden_dim,
                  })
    return config


class ParticleEmbedding(tf.keras.layers.Layer):
  """
  An embedding of CG or FG particles from their Cartesian coordinates to a new space.
  Geometric algebra attention (https://github.com/klarh/geometric_algebra_attention) is used to ensure
  that the embedding is permutation equivariant and rotation invariant.
  Note that translation invariance should also be applied by only passing local coordinates (relative to
  some reference site) to this function.
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
    Inputs:
        embedding_dim - final dimension of embedding; also used for working dimension of extra information
        hidden_dim - (40) dimension of all hidden dimensions in dense networks
        num_blocks - (2) number of attention blocks we will apply
        mask_zero - (True) whether or not to apply a mask layer to mask out zeros
    Outputs:
        ParticleEmbedding object
    """
    super(ParticleEmbedding, self).__init__(name=name, **kwargs)

    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.num_blocks = num_blocks
    self.mask_zero = mask_zero
    self.activation = activation

  def build(self, input_shape):

    #Network for mapping input extra info (particle identities,parameters, etc.) to working dimensionality
    self.info_net = tf.keras.layers.Dense(self.embedding_dim) #No activation, just linear mapping

    #Build attention blocks
    self.block_list = []
    for i in range(self.num_blocks):
      self.block_list.append(AttentionBlock(self.hidden_dim, activation=self.activation))

    if self.mask_zero:
      self.mask = tf.keras.layers.Masking()
    else:
      self.mask = None

    #Add one more attention layer with permutation invariance to sum over particles for embeddings
    #Consider a wrapper for just creating attention, since always need sequential models like this
    #Could then also use that in AttentionBlock, though there reduce=False, not True like here
    self.final_attn = gaa_keras.VectorAttention(tf.keras.models.Sequential([tf.keras.layers.Dense(self.hidden_dim, activation=self.activation),
                                                                            tf.keras.layers.Dense(1)
                                                                           ]),

                                                 tf.keras.models.Sequential([tf.keras.layers.Dense(self.hidden_dim),
                                                                             tf.keras.layers.LayerNormalization(),
                                                                             tf.keras.layers.Activation(self.activation),
                                                                             tf.keras.layers.Dense(self.embedding_dim)
                                                                           ]),
                                              reduce=True,
                                              merge_fun='concat',
                                              join_fun='concat',
                                              rank=3,
                                              )
  def call(self, coords, particle_info):
    """
    Inputs:
        coords - coordinates of particles
        particle_info - particle information like type or parameters
    Outputs:
        embedding - the embedding of these particles
    """
    if self.mask_zero:
      #Can only mask coordinates properly
      #Masking both paddings and any coordinates at origin (just CG bead we're backmapping, so ignore)
      coords = self.mask(coords)

    embedding = self.info_net(particle_info) #Won't be masked, but that's ok because gets masked in attention layer

    for block in self.block_list:
      embedding = block([coords, embedding])

    embedding = self.final_attn([coords, embedding])

    return embedding

  def get_config(self):
    config = super(AttentionBlock, self).get_config()
    config.update({"embedding_dim": self.embedding_dim,
                   "hidden_dim": self.hidden_dim,
                   "num_blocks": self.num_blocks,
                   "mask_zero": self.mask_zero,
                  })
    return config


class LocalParticleDescriptors(tf.keras.layers.Layer):
  """
  This class joins together distance masking and embedding layers to take Cartesian particle
  coordinates, mask them by distance to a reference while converting to local coordinates, and then
  compute embeddings, or descriptors. Likely, the mask and embed functions will be DistanceSelection
  and ParticleEmbedding objects. Mainly a wrapper for convenience since will use these functions
  together quite a bit.
  """

  def __init__(self,
               mask_fn,
               embed_fn,
               name='local_particle_desc',
               **kwargs):
    """
    Inputs:
        mask_fn - the distance-based masking function that also produces local coordinates of selected particles
        embed_fn - the embedding function to convert local coordinates to descriptors
    Outputs:
        LocalParticleDescriptors object
    """
    super(LocalParticleDescriptors, self).__init__(name=name, **kwargs)

    self.mask_fn = mask_fn
    self.embed_fn = embed_fn

  def call(self, coords, ref, props, box_lengths=None):
    """
    Inputs:
        coords - coordinates of particles to mask by distance (N_batch, (N_particles), 3);
                 note that this can be a ragged tensor where N_particles is different for each
                 batch index
        ref - coordinates of reference positions for determining mask (N_batch, 1, 3);
              only one reference coordinate for each batch element
        props - properties of particles, such as type, or parameters like charge (N_batch, (N_particles), N_props);
                can be ragged like coords, but first two dimensions, batch and number of particles, should match coords
        box_lengths - optional argument for box dimensions if box changes with given configuration and want to
                      wrap periodically when computing distances
    Outputs:
        descriptors - descriptors of the local environment around each reference position (N_batch, N_descriptors);
                      note that the number of descriptors is given by the output dimension of embed_fn
    """
    local_coords, local_props = self.mask_fn(coords, ref, particle_info=props, box_lengths=box_lengths)
    descriptors = self.embed_fn(local_coords, local_props)
    return descriptors

  def get_config(self):
    config = super(LocalParticleDescriptors, self).get_config()
    config.update({"mask_fn": self.mask_fn,
                   "embed_fn": self.embed_fn,
                  })
    return config

