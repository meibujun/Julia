import unittest
import numpy as np
from pyjwas.core import VarianceCovariance

class TestVarianceCovariance(unittest.TestCase):

    def test_creation_defaults(self):
        vc = VarianceCovariance()
        self.assertIsNone(vc.value)
        self.assertIsNone(vc.df)
        self.assertIsNone(vc.scale)
        self.assertTrue(vc.estimate_variance)
        self.assertFalse(vc.estimate_scale)
        self.assertFalse(vc.constraint)

    def test_creation_single_trait_scalar(self):
        vc = VarianceCovariance(value=10.5, df=4.0, scale=8.2,
                                estimate_variance=False, estimate_scale=True, constraint=False)
        self.assertEqual(vc.value, 10.5)
        self.assertEqual(vc.df, 4.0)
        self.assertEqual(vc.scale, 8.2)
        self.assertFalse(vc.estimate_variance)
        self.assertTrue(vc.estimate_scale)
        self.assertFalse(vc.constraint)

    def test_creation_multi_trait_numpy_array(self):
        val_arr = np.array([[10.0, 1.0], [1.0, 5.0]])
        scale_arr = np.array([[8.0, 0.5], [0.5, 4.0]])
        vc = VarianceCovariance(value=val_arr, df=5.0, scale=scale_arr, constraint=True)

        np.testing.assert_array_equal(vc.value, val_arr)
        self.assertEqual(vc.df, 5.0)
        np.testing.assert_array_equal(vc.scale, scale_arr)
        self.assertTrue(vc.estimate_variance) # Default
        self.assertFalse(vc.estimate_scale) # Default
        self.assertTrue(vc.constraint)

    def test_repr_method_scalar(self):
        vc = VarianceCovariance(value=10.5, df=4.0, scale=8.2)
        expected_repr = ("VarianceCovariance(value=10.5, df=4.0, scale=8.2, "
                         "estimate_variance=True, estimate_scale=False, constraint=False)")
        self.assertEqual(repr(vc), expected_repr)

    def test_repr_method_array(self):
        val_arr = np.array([[10.0, 1.0], [1.0, 5.0]])
        vc = VarianceCovariance(value=val_arr, df=5.0)
        # repr for value and scale shows shape if ndarray
        expected_repr_value = f"array({val_arr.shape})"
        expected_repr = (f"VarianceCovariance(value={expected_repr_value}, df=5.0, scale=None, "
                         f"estimate_variance=True, estimate_scale=False, constraint=False)")
        self.assertEqual(repr(vc), expected_repr)

if __name__ == '__main__':
    unittest.main()
