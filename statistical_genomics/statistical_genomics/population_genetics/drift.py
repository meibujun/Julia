import numpy as np

def simulate_genetic_drift(initial_allele_freq: float, population_size: int, num_generations: int) -> list[float]:
    """
    Simulates genetic drift for a single bi-allelic locus in a diploid population.

    Args:
        initial_allele_freq: The starting frequency of one allele (e.g., 'A').
                             Must be between 0.0 and 1.0, inclusive.
        population_size: The effective population size (N, number of diploid individuals).
                         Must be greater than 0.
        num_generations: The number of generations to simulate. Must be 0 or greater.

    Returns:
        A list of allele frequencies, one for each generation, starting with the
        initial frequency. The length of the list will be num_generations + 1.

    Raises:
        ValueError: If initial_allele_freq is not between 0 and 1.
        ValueError: If population_size is not greater than 0.
        ValueError: If num_generations is negative.
    """
    if not (0.0 <= initial_allele_freq <= 1.0):
        raise ValueError("Initial allele frequency must be between 0.0 and 1.0.")
    if population_size <= 0:
        raise ValueError("Population size must be greater than 0.")
    if num_generations < 0:
        raise ValueError("Number of generations cannot be negative.")

    allele_frequencies_over_time = [initial_allele_freq]
    current_freq = initial_allele_freq

    # Total number of alleles in the diploid population
    total_alleles_in_pop = 2 * population_size

    for _ in range(num_generations):
        if current_freq == 0.0 or current_freq == 1.0:
            # If allele is fixed or lost, it will remain so (no new mutations in this model)
            allele_frequencies_over_time.append(current_freq)
            continue

        # Number of alleles of interest (e.g., 'A') in the next generation
        # Drawn from a binomial distribution.
        # n = total number of alleles drawn for the next generation's gene pool
        # p = current frequency of the allele of interest
        num_alleles_of_interest = np.random.binomial(n=total_alleles_in_pop, p=current_freq)

        current_freq = num_alleles_of_interest / total_alleles_in_pop
        allele_frequencies_over_time.append(current_freq)

    return allele_frequencies_over_time

# Example usage:
# freqs = simulate_genetic_drift(initial_allele_freq=0.5, population_size=100, num_generations=50)
# print(freqs)
# import matplotlib.pyplot as plt
# plt.plot(freqs)
# plt.xlabel("Generation")
# plt.ylabel("Allele Frequency")
# plt.title("Genetic Drift Simulation")
# plt.ylim(0, 1)
# plt.show()
