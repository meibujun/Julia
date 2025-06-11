import unittest
import math
from statistical_genomics.statistical_genomics.stats.basic_stats import (
    calculate_mean,
    calculate_variance,
    calculate_std_dev
)

class TestBasicStats(unittest.TestCase):

    # Tests for calculate_mean
    def test_mean_simple(self):
        self.assertEqual(calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)
        self.assertAlmostEqual(calculate_mean([10.0, 20.0, 30.0]), 20.0)

    def test_mean_negative_numbers(self):
        self.assertEqual(calculate_mean([-1.0, -2.0, -3.0, -4.0, -5.0]), -3.0)
        self.assertAlmostEqual(calculate_mean([-10.0, 10.0, 0.0]), 0.0)

    def test_mean_empty_list(self):
        with self.assertRaisesRegex(ValueError, "Input list cannot be empty to calculate mean."):
            calculate_mean([])

    def test_mean_single_element(self):
        self.assertEqual(calculate_mean([42.0]), 42.0)
        self.assertEqual(calculate_mean([-7.5]), -7.5)

    # Tests for calculate_variance
    def test_variance_simple(self):
        # Variance of [1, 2, 3, 4, 5]
        # Mean = 3
        # Differences: -2, -1, 0, 1, 2
        # Squared differences: 4, 1, 0, 1, 4
        # Sum of squared differences = 10
        # Variance = 10 / (5-1) = 10 / 4 = 2.5
        self.assertAlmostEqual(calculate_variance([1.0, 2.0, 3.0, 4.0, 5.0]), 2.5)
        self.assertAlmostEqual(calculate_variance([2.0, 4.0, 6.0, 8.0]), 20.0/3.0) # Mean 5, Diffs: -3, -1, 1, 3. SqDiffs: 9,1,1,9. Sum=20. Var=20/3

    def test_variance_no_variance(self):
        self.assertAlmostEqual(calculate_variance([5.0, 5.0, 5.0, 5.0]), 0.0)
        self.assertAlmostEqual(calculate_variance([-2.0, -2.0, -2.0]), 0.0)

    def test_variance_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "Input list must contain at least two elements to calculate sample variance."):
            calculate_variance([])
        with self.assertRaisesRegex(ValueError, "Input list must contain at least two elements to calculate sample variance."):
            calculate_variance([1.0])

    # Tests for calculate_std_dev
    def test_std_dev_simple(self):
        self.assertAlmostEqual(calculate_std_dev([1.0, 2.0, 3.0, 4.0, 5.0]), math.sqrt(2.5))
        self.assertAlmostEqual(calculate_std_dev([2.0, 4.0, 6.0, 8.0]), math.sqrt(20.0/3.0))

    def test_std_dev_no_variance(self):
        self.assertAlmostEqual(calculate_std_dev([5.0, 5.0, 5.0, 5.0]), 0.0)
        self.assertAlmostEqual(calculate_std_dev([-2.0, -2.0, -2.0]), 0.0)

    def test_std_dev_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "Input list must contain at least two elements to calculate sample standard deviation."):
            calculate_std_dev([])
        with self.assertRaisesRegex(ValueError, "Input list must contain at least two elements to calculate sample standard deviation."):
            calculate_std_dev([1.0])

    def test_std_dev_matches_variance(self):
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        var1 = calculate_variance(data1)
        std_dev1 = calculate_std_dev(data1)
        self.assertAlmostEqual(std_dev1 ** 2, var1)

        data2 = [10.0, 15.0, 12.0, 18.0, 20.0]
        var2 = calculate_variance(data2)
        std_dev2 = calculate_std_dev(data2)
        self.assertAlmostEqual(std_dev2 ** 2, var2)

        # Test with zero variance
        data3 = [7.0, 7.0, 7.0]
        var3 = calculate_variance(data3)
        std_dev3 = calculate_std_dev(data3)
        self.assertAlmostEqual(std_dev3 ** 2, var3) # 0.0 == 0.0

if __name__ == '__main__':
    unittest.main()
