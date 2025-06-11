import unittest
import numpy as np # numpy is used by the function being tested
from statistical_genomics.statistical_genomics.population_genetics.drift import simulate_genetic_drift

class TestGeneticDrift(unittest.TestCase):

    def test_drift_input_validation(self):
        # Test initial_allele_freq out of range
        with self.assertRaisesRegex(ValueError, "Initial allele frequency must be between 0.0 and 1.0."):
            simulate_genetic_drift(initial_allele_freq=-0.1, population_size=100, num_generations=10)
        with self.assertRaisesRegex(ValueError, "Initial allele frequency must be between 0.0 and 1.0."):
            simulate_genetic_drift(initial_allele_freq=1.1, population_size=100, num_generations=10)

        # Test population_size <= 0
        with self.assertRaisesRegex(ValueError, "Population size must be greater than 0."):
            simulate_genetic_drift(initial_allele_freq=0.5, population_size=0, num_generations=10)
        with self.assertRaisesRegex(ValueError, "Population size must be greater than 0."):
            simulate_genetic_drift(initial_allele_freq=0.5, population_size=-10, num_generations=10)

        # Test num_generations < 0
        with self.assertRaisesRegex(ValueError, "Number of generations cannot be negative."):
            simulate_genetic_drift(initial_allele_freq=0.5, population_size=100, num_generations=-1)

    def test_drift_output_length(self):
        self.assertEqual(len(simulate_genetic_drift(0.5, 100, 0)), 1) # Initial freq only
        self.assertEqual(len(simulate_genetic_drift(0.5, 100, 1)), 2)
        self.assertEqual(len(simulate_genetic_drift(0.5, 100, 10)), 11)
        self.assertEqual(len(simulate_genetic_drift(0.2, 50, 50)), 51)

    def test_drift_output_values_range(self):
        np.random.seed(42) # for reproducibility of stochastic part
        freqs = simulate_genetic_drift(initial_allele_freq=0.5, population_size=20, num_generations=50)
        for freq in freqs:
            self.assertTrue(0.0 <= freq <= 1.0, f"Frequency {freq} is out of [0,1] range.")

    def test_drift_fixation_or_loss_deterministic_cases(self):
        # Test that for initial_allele_freq=0.0, all subsequent frequencies are 0.0.
        freqs_at_zero = simulate_genetic_drift(initial_allele_freq=0.0, population_size=50, num_generations=100)
        for freq in freqs_at_zero:
            self.assertEqual(freq, 0.0, "Frequency should remain 0.0 if starting at 0.0")

        # Test that for initial_allele_freq=1.0, all subsequent frequencies are 1.0.
        freqs_at_one = simulate_genetic_drift(initial_allele_freq=1.0, population_size=50, num_generations=100)
        for freq in freqs_at_one:
            self.assertEqual(freq, 1.0, "Frequency should remain 1.0 if starting at 1.0")

    def test_drift_stochastic_outcomes_N1_G1(self):
        # With N=1, p=0.5, G=1:
        # Total alleles = 2*N = 2.
        # np.random.binomial(n=2, p=0.5) can give 0, 1, or 2.
        # Next freqs can be 0/2=0.0, 1/2=0.5, 2/2=1.0.
        # We can't assert one specific outcome, but we can run it many times or check properties.
        # For this test, run once and check if it's one of the possibilities.
        # Seeding makes this particular run deterministic.
        np.random.seed(123)
        freqs = simulate_genetic_drift(initial_allele_freq=0.5, population_size=1, num_generations=1)
        self.assertEqual(len(freqs), 2)
        self.assertIn(freqs[1], [0.0, 0.5, 1.0],
                      f"For N=1, p=0.5, G=1, next freq {freqs[1]} was not in [0.0, 0.5, 1.0]")

        # Another seed to try to get a different result (though not guaranteed)
        np.random.seed(124)
        freqs_s2 = simulate_genetic_drift(initial_allele_freq=0.5, population_size=1, num_generations=1)
        self.assertIn(freqs_s2[1], [0.0, 0.5, 1.0])

        np.random.seed(125)
        freqs_s3 = simulate_genetic_drift(initial_allele_freq=0.5, population_size=1, num_generations=1)
        self.assertIn(freqs_s3[1], [0.0, 0.5, 1.0])


    def test_drift_initial_frequency_correct(self):
        initial_freq = 0.3
        freqs = simulate_genetic_drift(initial_allele_freq=initial_freq, population_size=100, num_generations=10)
        self.assertEqual(freqs[0], initial_freq, "First element in result list should be the initial frequency.")

    def test_drift_zero_generations(self):
        initial_freq = 0.75
        freqs = simulate_genetic_drift(initial_allele_freq=initial_freq, population_size=50, num_generations=0)
        self.assertEqual(len(freqs), 1, "List should have 1 element for 0 generations.")
        self.assertEqual(freqs[0], initial_freq, "The only element should be the initial frequency.")

if __name__ == '__main__':
    unittest.main()
