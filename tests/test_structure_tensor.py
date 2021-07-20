import unittest

import numpy as np
from structure_tensor import (eig_special_2d, eig_special_3d,
                              structure_tensor_2d, structure_tensor_3d, util)


class TestStructureTensor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_smoke_2d(self):
        """Test functions don't crash."""
        S = structure_tensor_2d(np.random.random((64, 64)), 2.5, 1.5)
        val, vec = eig_special_2d(S)
        self.assertTrue(True)

    def test_smoke_3d(self):
        """Test functions don't crash."""
        S = structure_tensor_3d(np.random.random((64, 64, 64)), 2.5, 1.5)
        val, vec = eig_special_3d(S)
        self.assertTrue(True)

    def test_cupy_missing(self):
        """Can't test CuPy on machine without CUDA."""
        try:
            from structure_tensor.cp import eig_special_3d, structure_tensor_3d
            self.assertTrue(True, 'CuPy available')
        except Exception as ex:
            self.assertEqual(ex.msg, "No module named 'cupy'")
            return

        S = structure_tensor_3d(np.random.random((64, 64, 64)), 2.5, 1.5)
        val, vec = eig_special_3d(S)
        self.assertTrue(True)


class TestUtil(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.volume = np.random.random((128, 128, 50))
        self.sigma = 2
        self.truncate = 4.0
        self.block_size = 50
        self.kernel_radius = int(self.sigma * self.truncate + 0.5)

    def assert_block(self, b, pos, pad):
        """Assertions for block, position and padding."""
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(pos, np.ndarray)
        self.assertIsInstance(pad, np.ndarray)
        self.assertGreater(b.size, 0)
        self.assertEqual(pos.size, 6)
        self.assertEqual(pad.size, 6)

    def test_get_block_generator(self):
        """Test block generator function."""
        for b, pos, pad in util.get_block_generator(self.volume, self.sigma,
                                                    self.block_size,
                                                    self.truncate):
            self.assert_block(b, pos, pad)

    def test_get_blocks(self):
        """Test blocks function."""
        blocks, positions, paddings = util.get_blocks(self.volume, self.sigma,
                                                      self.block_size,
                                                      self.truncate)
        self.assertEqual(len(blocks), len(positions))
        self.assertEqual(len(blocks), len(paddings))
        for b, pos, pad in zip(blocks, positions, paddings):
            self.assert_block(b, pos, pad)

    def test_get_block_count(self):
        """Test block count function."""
        blocks, positions, paddings = util.get_blocks(self.volume, self.sigma,
                                                      self.block_size,
                                                      self.truncate)
        block_count = util.get_block_count(self.volume, self.block_size)
        self.assertEqual(len(blocks), block_count)

    def test_get_block(self):
        """Test get ith block function."""
        for i, (b1, pos1, pad1) in enumerate(
                util.get_block_generator(self.volume, self.sigma,
                                         self.block_size, self.truncate)):

            b2, pos2, pad2 = util.get_block(i, self.volume, self.sigma,
                                            self.block_size, self.truncate)

            self.assert_block(b2, pos2, pad2)

            np.testing.assert_array_equal(b1, b2)
            np.testing.assert_array_equal(pos1, pos2)
            np.testing.assert_array_equal(pad1, pad2)

    def test_remove_padding(self):
        """Test the remove padding function."""
        for b, pos, pad in util.get_block_generator(self.volume, self.sigma,
                                                    self.block_size,
                                                    self.truncate):
            b2 = util.remove_padding(b, pad)
            for i in range(len(pad)):
                self.assertEqual(b.shape[i] - np.sum(pad[i]), b2.shape[i])

    def test_remove_boundary(self):
        """Test the remove boundary function."""
        for b, pos, pad in util.get_block_generator(self.volume, self.sigma,
                                                    self.block_size):
            b2 = util.remove_boundary(b, pad, self.sigma)
            for i in range(len(pad)):
                self.assertEqual(
                    b.shape[i] -
                    np.sum(np.maximum(0, self.kernel_radius - pad[i])),
                    b2.shape[i])

    def test_insert_block(self):
        """Test the insert block function."""
        vol = np.zeros_like(self.volume)
        for b, pos, pad in util.get_block_generator(self.volume, self.sigma,
                                                    self.block_size,
                                                    self.truncate):
            util.insert_block(vol, b, pos, pad)

        np.testing.assert_array_equal(self.volume, vol)

        positions_with_values = []

        for b, pos, pad in util.get_block_generator(self.volume, self.sigma,
                                                    self.block_size,
                                                    self.truncate):
            b = util.remove_padding(b, pad)

            mask = np.zeros(b.shape, dtype=bool)
            mask[:, 0, 0] = True
            mask[0, :, 0] = True
            mask[0, 0, :] = True

            util.insert_block(vol, b[mask], pos, pad=None, mask=mask)

            positions_with_values.append(pos[:, 0])

        positions_with_values = np.asarray(positions_with_values)
        positions_with_values = tuple(positions_with_values.transpose())

        np.testing.assert_array_equal(self.volume[positions_with_values],
                                      vol[positions_with_values])


if __name__ == '__main__':
    unittest.main()
