"""
Tests for mappings module.
"""

import numpy as np
import tensorflow as tf

import pytest

from vaemolsim import mappings


class TestFCDeepNN:

    def test_default_creation(self, normal_sample):
        nn = mappings.FCDeepNN(3)
        assert nn.target_shape == (3, )
        # And build it by passing input through
        out = nn(normal_sample)
        assert len(nn.periodic_dofs) == np.prod(normal_sample.shape[1:])
        assert len(nn.layer_list) == len(nn.hidden_dim) + 2  # No batch normalization
        assert out.shape == np.hstack([normal_sample.shape[0], nn.target_shape])

    def test_batch_norm(self, normal_sample):
        nn = mappings.FCDeepNN(3, batch_norm=True)
        _ = nn(normal_sample)
        assert len(nn.layer_list) == len(nn.hidden_dim) * 2 + 2

    def test_multi_hidden(self, normal_sample):
        nn = mappings.FCDeepNN(3, hidden_dim=[20, 15, 10])
        _ = nn(normal_sample)
        assert len(nn.hidden_dim) == 3
        assert len(nn.layer_list) == len(nn.hidden_dim) + 2

    def test_periodic_dofs(self, normal_sample):
        nn = mappings.FCDeepNN(3,
                               periodic_dofs=np.round(np.random.random(np.prod(normal_sample.shape[1:]))).astype(bool))
        out = nn(normal_sample)
        assert np.any(nn.periodic_dofs) == nn.any_periodic
        assert out.shape == np.hstack([normal_sample.shape[0], nn.target_shape])
        assert (nn.layer_list[0].weights[0].shape == (np.prod(normal_sample.shape[1:]) + np.sum(nn.periodic_dofs),
                                                      nn.hidden_dim[0]))

    def test_flatten(self):
        inputs = tf.random.uniform((4, 6, 3))
        nn = mappings.FCDeepNN(2)
        out = nn(inputs)
        assert out.shape == np.hstack([inputs.shape[0], nn.target_shape])

    def test_multidim_shape(self, normal_sample):
        nn = mappings.FCDeepNN((3, 2))
        out = nn(normal_sample)
        assert out.shape == np.hstack([normal_sample.shape[0], nn.target_shape])


class TestDistanceSelection:

    coords = tf.random.uniform((4, 100, 3), -5.0, 5.0)
    ref = tf.random.uniform((4, 1, 3), -5.0, 5.0)
    box_l = np.array([10.0, 10.0, 10.0], dtype='float32')

    def test_default_creation(self):
        md = mappings.DistanceSelection(3.0)
        assert md.sq_cut == 9.0
        sel = md(self.coords, self.ref)
        assert sel.shape == (self.ref.shape[0], md.max_included, 3)
        box_sel = md(self.coords, self.ref, box_lengths=np.tile(self.box_l, (self.ref.shape[0], 1)))
        assert box_sel.shape == (self.ref.shape[0], md.max_included, 3)
        assert not np.all(sel == box_sel)

    def test_max_included(self):
        md = mappings.DistanceSelection(3.0, max_included=5)
        sel = md(self.coords, self.ref)
        assert sel.shape == (self.ref.shape[0], 5, 3)

    def test_box_lengths(self):
        md = mappings.DistanceSelection(3.0, box_lengths=self.box_l)
        sel = md(self.coords, self.ref)
        box_sel = md(self.coords, self.ref, box_lengths=np.tile(self.box_l, (self.ref.shape[0], 1)))
        np.testing.assert_allclose(sel, box_sel)

    def test_particle_info(self):
        particle_info = tf.round(tf.random.uniform(np.hstack([self.coords.shape[:2], 7])))
        md = mappings.DistanceSelection(3.0)
        sel, info = md(self.coords, self.ref, particle_info=particle_info)
        assert info.shape == (self.ref.shape[0], md.max_included, particle_info.shape[-1])

    def test_ragged(self):
        rag_shapes = [1, 0, 5, 2]
        cr = tf.RaggedTensor.from_row_splits(np.vstack([np.random.uniform(-5.0, 5.0, (n, 3)) for n in rag_shapes]),
                                             row_splits=np.hstack([0, np.cumsum(rag_shapes)]))
        cr_info = tf.RaggedTensor.from_row_splits(np.vstack(
            [np.round(np.random.uniform(0.0, 1.0, (n, 7))) for n in rag_shapes]),
                                                  row_splits=np.hstack([0, np.cumsum(rag_shapes)]))
        md = mappings.DistanceSelection(3.0, box_lengths=self.box_l)
        sel, info = md(cr, self.ref, particle_info=cr_info)
        assert sel.shape == (self.ref.shape[0], md.max_included, 3)
        assert info.shape == (self.ref.shape[0], md.max_included, cr_info.shape[-1])


class TestAttentionBlock:

    coords = tf.random.uniform((4, 6, 3), -5.0, 5.0)
    c_info = tf.round(tf.random.uniform((4, 6, 9), 0.0, 1.0))

    def test_default_creation(self):
        attn = mappings.AttentionBlock()
        out = attn([self.coords, self.c_info])
        assert out.shape == self.c_info.shape

    def test_hidden_dim(self):
        attn = mappings.AttentionBlock(hidden_dim=20)
        assert attn.hidden_dim == 20
        out = attn([self.coords, self.c_info])
        assert out.shape == self.c_info.shape


class TestParticleEmbedding:

    coords = tf.random.uniform((4, 100, 3), -5.0, 5.0)
    c_info = tf.round(tf.random.uniform((4, 100, 9), 0.0, 1.0))

    def test_default_creation(self):
        pe = mappings.ParticleEmbedding(20)
        assert pe.mask_zero
        out = pe(self.coords, self.c_info)
        assert out.shape == (self.coords.shape[0], pe.embedding_dim)
        assert isinstance(pe.mask, tf.keras.layers.Masking)
        assert len(pe.block_list) == pe.num_blocks

    def test_masking(self):
        ref = tf.random.uniform((4, 1, 3), -5.0, 5.0)
        md = mappings.DistanceSelection(3.0, box_lengths=[10.0, 10.0, 10.0])
        c_mapped, c_info_mapped = md(self.coords, ref, particle_info=self.c_info)
        no_mask_pe = mappings.ParticleEmbedding(20, mask_zero=False)
        assert not no_mask_pe.mask_zero
        out_no = no_mask_pe(c_mapped, c_info_mapped)
        assert out_no.shape == (c_mapped.shape[0], no_mask_pe.embedding_dim)
        assert no_mask_pe.mask is None
        mask_pe = mappings.ParticleEmbedding(20, mask_zero=True)
        assert mask_pe.mask_zero
        out_yes = mask_pe(c_mapped, c_info_mapped)
        assert out_yes.shape == (c_mapped.shape[0], mask_pe.embedding_dim)
        assert out_no.shape == out_yes.shape
        assert not np.all(out_no == out_yes)
        with pytest.raises(AttributeError, match='_keras_mask'):
            _ = out_no._keras_mask
        assert hasattr(out_yes, '_keras_mask')


def test_localparticledescriptors():
    c = tf.random.uniform((4, 100, 3), -5.0, 5.0)
    ref = tf.random.uniform((4, 1, 3), -5.0, 5.0)
    c_info = tf.round(tf.random.uniform((4, 100, 9), 0.0, 1.0))
    map_dist = mappings.DistanceSelection(3.0, max_included=10, box_lengths=[10.0, 10.0, 10.0])
    pe = mappings.ParticleEmbedding(20)
    map_and_embed = mappings.LocalParticleDescriptors(map_dist, pe)
    out = map_and_embed(c, ref, c_info)
    assert out.shape == (c.shape[0], pe.embedding_dim)
