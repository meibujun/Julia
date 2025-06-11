import unittest
import math # For math.isclose or direct float comparison if needed, though assertAlmostEqual is better
import scipy.stats # For calculating expected slope in scattered data test
import numpy as np # For creating arrays for the scattered data test's expected calculation
from statistical_genomics.statistical_genomics.quantitative_genetics.heritability import estimate_narrow_sense_heritability_po_regression

class TestHeritability(unittest.TestCase):

    def test_heritability_input_validation(self):
        # Test if lists are different lengths
        with self.assertRaisesRegex(ValueError, "Parent and offspring value lists must have the same length."):
            estimate_narrow_sense_heritability_po_regression([1.0, 2.0], [1.0])
        with self.assertRaisesRegex(ValueError, "Parent and offspring value lists must have the same length."):
            estimate_narrow_sense_heritability_po_regression([1.0], [1.0, 2.0])

        # Test if lists are empty
        with self.assertRaisesRegex(ValueError, "Input lists .* cannot be empty."):
            estimate_narrow_sense_heritability_po_regression([], [])
        with self.assertRaisesRegex(ValueError, "Input lists .* cannot be empty."):
            estimate_narrow_sense_heritability_po_regression([1.0, 2.0], [])
        with self.assertRaisesRegex(ValueError, "Input lists .* cannot be empty."):
            estimate_narrow_sense_heritability_po_regression([], [1.0, 2.0])

        # Test if lists have fewer than 2 elements
        with self.assertRaisesRegex(ValueError, "At least two data points are required for regression analysis."):
            estimate_narrow_sense_heritability_po_regression([1.0], [1.0])

    def test_heritability_perfect_heritability(self):
        parents = [1.0, 2.0, 3.0, 4.0, 5.0]
        offspring = [1.0, 2.0, 3.0, 4.0, 5.0] # y = 1*x + 0
        h2 = estimate_narrow_sense_heritability_po_regression(parents, offspring)
        self.assertAlmostEqual(h2, 1.0, places=7)

    def test_heritability_half_heritability(self):
        parents = [1.0, 2.0, 3.0, 4.0, 5.0]
        offspring = [0.5, 1.0, 1.5, 2.0, 2.5] # y = 0.5*x + 0
        h2 = estimate_narrow_sense_heritability_po_regression(parents, offspring)
        self.assertAlmostEqual(h2, 0.5, places=7)

        # With an intercept
        offspring_intercept = [1.5, 2.0, 2.5, 3.0, 3.5] # y = 0.5*x + 1
        h2_intercept = estimate_narrow_sense_heritability_po_regression(parents, offspring_intercept)
        self.assertAlmostEqual(h2_intercept, 0.5, places=7)


    def test_heritability_zero_heritability(self):
        parents = [1.0, 2.0, 3.0, 4.0, 5.0]
        offspring = [3.0, 3.0, 3.0, 3.0, 3.0] # y = 0*x + 3
        h2 = estimate_narrow_sense_heritability_po_regression(parents, offspring)
        self.assertAlmostEqual(h2, 0.0, places=7)

    def test_heritability_scattered_data(self):
        parents = [1.0, 2.0, 3.0, 4.0, 5.0]
        offspring = [1.1, 1.9, 3.2, 3.8, 5.3] # Data points

        # Calculate expected slope using scipy.stats.linregress directly for comparison
        expected_slope, _, _, _, _ = scipy.stats.linregress(np.array(parents), np.array(offspring))

        h2 = estimate_narrow_sense_heritability_po_regression(parents, offspring)
        self.assertAlmostEqual(h2, expected_slope, places=7)
        # For this specific data:
        # X = [1, 2, 3, 4, 5], Y = [1.1, 1.9, 3.2, 3.8, 5.3]
        # Mean_X = 3, Mean_Y = 3.06
        # Sum( (X-MeanX)(Y-MeanY) ) = (-2*-1.96) + (-1*-1.16) + (0*0.14) + (1*0.74) + (2*2.24)
        # = 3.92 + 1.16 + 0 + 0.74 + 4.48 = 10.3
        # Sum( (X-MeanX)^2 ) = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 4 + 1 + 0 + 1 + 4 = 10
        # Slope = 10.3 / 10 = 1.03
        self.assertAlmostEqual(h2, 1.03, places=7)


    def test_heritability_negative_correlation(self):
        parents = [1.0, 2.0, 3.0, 4.0, 5.0]
        offspring = [5.0, 4.0, 3.0, 2.0, 1.0] # y = -1*x + 6
        h2 = estimate_narrow_sense_heritability_po_regression(parents, offspring)
        # Heritability (h^2) is usually defined between 0 and 1.
        # A negative slope from parent-offspring regression technically means h^2 < 0,
        # which is biologically unexpected for a simple additive model but mathematically possible.
        # The function returns the slope, so we test for that.
        self.assertAlmostEqual(h2, -1.0, places=7)

if __name__ == '__main__':
    unittest.main()
