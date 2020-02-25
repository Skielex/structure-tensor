import unittest

import numpy as np

from structure_tensor import eig_special_2d, eig_special_3d, structure_tensor_2d, structure_tensor_3d


class TestGraph(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
