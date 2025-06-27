import unittest
import numpy as np
# Adjust the import path based on how you run tests (e.g., from project root)
# This assumes tests are run in a way that pyjwas is in PYTHONPATH
from pyjwas.core import ModelTerm

class TestModelTerm(unittest.TestCase):

    def test_creation_basic(self):
        term = ModelTerm(term_str="age", model_index=1, trait_name="y1")
        self.assertEqual(term.imodel, 1)
        self.assertEqual(term.itrait, "y1")
        self.assertEqual(term.trm_str, "y1:age")
        self.assertEqual(term.n_factors, 1)
        self.assertEqual(term.factors, ["age"])
        self.assertEqual(term.random_type, "fixed")
        self.assertEqual(term.n_levels, 0) # Default before data processing
        self.assertEqual(term.start_pos, 0) # Default
        self.assertIsNone(term.val)
        self.assertIsNone(term.X)
        self.assertEqual(term.data, [])
        self.assertEqual(term.names, [])

    def test_creation_interaction_term(self):
        term = ModelTerm(term_str="herd * year", model_index=2, trait_name="milk_yield")
        self.assertEqual(term.imodel, 2)
        self.assertEqual(term.itrait, "milk_yield")
        self.assertEqual(term.trm_str, "milk_yield:herd * year")
        self.assertEqual(term.n_factors, 2)
        self.assertEqual(term.factors, ["herd", "year"]) # Constructor splits by '*'

    def test_creation_with_stripping(self):
        term = ModelTerm(term_str="  age  ", model_index=1, trait_name="  y1  ")
        self.assertEqual(term.itrait, "y1")
        self.assertEqual(term.trm_str, "y1:age") # Note: Julia code has specific splitting for factors
        self.assertEqual(term.factors, ["age"])


    def test_attribute_modification(self):
        term = ModelTerm(term_str="sex", model_index=1, trait_name="y1")
        term.n_levels = 2
        term.names = ["Male", "Female"]
        term.random_type = "factor"
        term.start_pos = 5
        dummy_X = np.array([[1, 0], [0, 1], [1, 0]])
        term.X = dummy_X
        term.val = np.array([1.0, 1.0, 1.0])
        term.data = ["Male", "Female", "Male"]

        self.assertEqual(term.n_levels, 2)
        self.assertEqual(term.names, ["Male", "Female"])
        self.assertEqual(term.random_type, "factor")
        self.assertEqual(term.start_pos, 5)
        np.testing.assert_array_equal(term.X, dummy_X)
        np.testing.assert_array_equal(term.val, np.array([1.0, 1.0, 1.0]))
        self.assertEqual(term.data, ["Male", "Female", "Male"])

    def test_repr_method(self):
        term = ModelTerm(term_str="age", model_index=1, trait_name="y1")
        term.n_levels = 1
        term.start_pos = 3
        expected_repr = "ModelTerm(trm_str='y1:age', imodel=1, itrait='y1', n_factors=1, factors=['age'], random_type='fixed', n_levels=1, start_pos=3)"
        self.assertEqual(repr(term), expected_repr)

if __name__ == '__main__':
    unittest.main()
