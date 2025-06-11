import numpy as np
from scipy import stats

def calculate_allele_frequencies(genotypes: dict[str, int]) -> dict[str, float]:
    """
    Calculates allele frequencies from genotype counts.

    Assumes diploid organisms and two alleles per locus.
    Genotypes are expected in formats like "AA", "Aa", "aa".

    Args:
        genotypes (dict[str, int]): A dictionary where keys are genotypes
                                     (e.g., "AA", "Aa", "aa") and values are
                                     their counts.

    Returns:
        dict[str, float]: A dictionary with alleles (e.g., "A", "a") and
                          their frequencies.

    Raises:
        ValueError: If the genotypes dictionary is empty, contains invalid
                    genotype strings, or implies more than two alleles.
    """
    if not genotypes:
        raise ValueError("Genotypes dictionary cannot be empty.")

    allele_counts = {}
    total_individuals = sum(genotypes.values())

    if total_individuals == 0:
        # Return zero frequencies if there are no individuals
        # Determine alleles from genotype keys
        all_alleles = set()
        for genotype_str in genotypes.keys():
            if len(genotype_str) != 2:
                raise ValueError(f"Invalid genotype string: {genotype_str}. Expected 2 characters.")
            all_alleles.add(genotype_str[0])
            all_alleles.add(genotype_str[1])
        return {allele: 0.0 for allele in sorted(list(all_alleles))}


    for genotype_str, count in genotypes.items():
        if len(genotype_str) != 2:
            raise ValueError(f"Invalid genotype string: {genotype_str}. Expected 2 characters.")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Count for genotype {genotype_str} must be a non-negative integer.")

        allele1, allele2 = genotype_str[0], genotype_str[1]
        allele_counts[allele1] = allele_counts.get(allele1, 0) + count
        allele_counts[allele2] = allele_counts.get(allele2, 0) + count

    if len(allele_counts) > 2 and total_individuals > 0 : # Allow more than 2 if counts are 0
        # Check if more than 2 alleles have actual counts
        positive_allele_counts = sum(1 for count in allele_counts.values() if count > 0)
        if positive_allele_counts > 2:
            raise ValueError("More than two alleles detected. This function currently supports only two alleles per locus.")


    total_alleles = 2 * total_individuals
    if total_alleles == 0 and not allele_counts: # No individuals, no genotypes
         return {} # Or raise error, depends on desired behavior for empty input with no genotype structure
    if total_alleles == 0 and allele_counts: # Genotypes like {"AA":0, "AB":0}, but no individuals
        return {allele: 0.0 for allele in allele_counts}


    allele_frequencies = {
        allele: count / total_alleles for allele, count in allele_counts.items()
    }
    return allele_frequencies


def calculate_genotype_frequencies(genotypes: dict[str, int]) -> dict[str, float]:
    """
    Calculates genotype frequencies from genotype counts.

    Args:
        genotypes (dict[str, int]): A dictionary of genotype counts.

    Returns:
        dict[str, float]: A dictionary of genotype frequencies.

    Raises:
        ValueError: If the genotypes dictionary is empty or contains invalid counts.
    """
    if not genotypes:
        raise ValueError("Genotypes dictionary cannot be empty.")

    total_individuals = sum(genotypes.values())
    if total_individuals == 0:
        return {gt: 0.0 for gt in genotypes}

    for genotype_str, count in genotypes.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Count for genotype {genotype_str} must be a non-negative integer.")


    genotype_frequencies = {
        gt: count / total_individuals for gt, count in genotypes.items()
    }
    return genotype_frequencies


def check_hardy_weinberg_equilibrium(
    observed_genotypes: dict[str, int]
) -> tuple[bool, float, dict[str, float]]:
    """
    Checks if a population is in Hardy-Weinberg Equilibrium (HWE).

    Args:
        observed_genotypes (dict[str, int]): A dictionary of observed
                                             genotype counts (e.g., {"AA": 25, "Aa": 50, "aa": 25}).

    Returns:
        tuple[bool, float, dict[str, float]]: A tuple containing:
            - bool: True if in HWE (Chi-squared p-value > 0.05), False otherwise.
            - float: The Chi-squared test statistic.
            - dict[str, float]: Expected genotype frequencies under HWE.

    Raises:
        ValueError: If input is invalid (e.g., not enough genotypes, zero individuals).
    """
    if not observed_genotypes or len(observed_genotypes) < 2 : # Need at least two genotypes for HWE
        raise ValueError(
            "Observed genotypes dictionary must contain at least two genotypes."
        )

    total_observed = sum(observed_genotypes.values())
    if total_observed == 0:
        raise ValueError("Total number of observed individuals cannot be zero.")

    # Ensure correct genotype format "AA", "Aa", "aa" etc.
    # And identify the alleles present.
    alleles = set()
    alleles = set()
    processed_genotypes = {} # Store standardized genotypes and their counts

    for gt, count in observed_genotypes.items():
        if len(gt) != 2:
            raise ValueError(f"Invalid genotype string: {gt}. Expected 2 characters.")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Count for genotype {gt} must be a non-negative integer.")

        alleles.add(gt[0])
        alleles.add(gt[1])

        if gt[0] == gt[1]: # Homozygous
            processed_genotypes[gt] = processed_genotypes.get(gt, 0) + count
        else: # Heterozygous
            sorted_gt = "".join(sorted(gt)) # Standardize (e.g., bA to Ab)
            processed_genotypes[sorted_gt] = processed_genotypes.get(sorted_gt, 0) + count

    if len(alleles) != 2:
        raise ValueError("Hardy-Weinberg equilibrium check requires exactly two alleles.")

    allele1, allele2 = sorted(list(alleles)) # e.g., A, a

    # Define the three canonical genotype strings based on sorted alleles
    gt_hom1 = allele1 * 2  # e.g., AA
    gt_het = allele1 + allele2 # e.g., Aa
    gt_hom2 = allele2 * 2  # e.g., aa

    # Ensure all three expected genotypes are present in processed_genotypes, adding if count is 0
    # This is important for calculate_allele_frequencies and for chi-squared categories
    final_genotype_counts = {
        gt_hom1: processed_genotypes.get(gt_hom1, 0),
        gt_het: processed_genotypes.get(gt_het, 0),
        gt_hom2: processed_genotypes.get(gt_hom2, 0)
    }

    # Recalculate total_observed based on the counts of only these three genotypes
    # This handles cases where original observed_genotypes might have had extraneous entries
    # that were filtered out by the allele check or genotype standardization.
    # However, total_observed was calculated at the start from the raw input.
    # If final_genotype_counts sum is different, it means some input genotypes were irrelevant
    # or malformed (e.g. "AB" "BC" "CA" which has 3 alleles).
    # The initial total_observed is still the correct denominator for allele frequency calculation
    # if we consider all individuals contributing to allele pool.
    # But for HWE chi-square, we need counts of the 3 specific genotypes.

    # Calculate allele frequencies using the (potentially standardized) final_genotype_counts
    # The calculate_allele_frequencies function expects raw counts of AA, Aa, aa.
    current_allele_freqs = calculate_allele_frequencies(final_genotype_counts)
    p = current_allele_freqs.get(allele1, 0.0)
    q = current_allele_freqs.get(allele2, 0.0)

    # If total_observed from the start is 0, it's already handled.
    # If after processing, the sum of final_genotype_counts is 0 (e.g. {"AA":0, "Aa":0, "aa":0}),
    # but total_observed was >0 (e.g. from {"AB":10}), this is an inconsistent state.
    # The initial total_observed should be the sum of counts from the *original* dictionary.
    # The chi-squared test should be on the counts of the three specific genotypes.

    observed_total_for_chi2 = sum(final_genotype_counts.values())
    if observed_total_for_chi2 == 0 :
        # This case implies that the genotypes relevant to the two chosen alleles (e.g. AA, Aa, aa)
        # all have zero counts. If the original total_observed was also zero, it's fine (handled earlier).
        # If original total_observed > 0, but these specific genotype counts are 0,
        # then HWE cannot be meaningfully calculated for these specific forms.
        # Example: observed_genotypes = {"CC": 10, "AA":0, "Aa":0, "aa":0}. Alleles A,a.
        # This scenario is complex. For now, if relevant counts are zero, chi2 is undefined.
        # However, calculate_allele_frequencies would return p=0, q=0 or similar if only AA,Aa,aa are keys.
        # Let's rely on the initial total_observed check. If that passed, this sum should also be >0
        # unless the input was like {"CC":10} and we are checking for A,a.
        # The current logic of calculate_allele_frequencies will also use the sum of values
        # from the dictionary passed to it.
         raise ValueError("Sum of counts for the expected genotypes (e.g. AA, Aa, aa) is zero, cannot perform HWE.")


    expected_counts = {
        gt_hom1: (p**2) * observed_total_for_chi2,
        gt_het: (2 * p * q) * observed_total_for_chi2,
        gt_hom2: (q**2) * observed_total_for_chi2,
    }

    observed_values = [
        final_genotype_counts[gt_hom1],
        final_genotype_counts[gt_het],
        final_genotype_counts[gt_hom2],
    ]
    expected_values = [
        expected_counts[gt_hom1],
        expected_counts[gt_het],
        expected_counts[gt_hom2],
    ]

    # Chi-squared test
    # Degrees of freedom = number of categories - 1 - number of parameters estimated
    # Here, 3 categories (genotypes), 1 parameter estimated (p, since q=1-p) => df = 3 - 1 - 1 = 1
    # ddof = degrees of freedom consumed by estimating parameters from the data.
    # We estimate one parameter (p, as q=1-p) from the data to calculate expected values.
    chi2_stat, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values, ddof=1)

    # Expected genotype frequencies
    expected_genotype_freqs = {
        gt: count / observed_total_for_chi2 if observed_total_for_chi2 > 0 else 0
        for gt, count in expected_counts.items()
    }

    # Significance level, typically 0.05
    alpha = 0.05
    is_in_hwe = p_value > alpha

    return is_in_hwe, chi2_stat, expected_genotype_freqs


def calculate_phenotypic_variance(phenotypes: list[float]) -> float:
    """
    Calculates the phenotypic variance from a list of phenotypic values.

    Args:
        phenotypes (list[float]): A list of phenotypic values.

    Returns:
        float: The phenotypic variance.

    Raises:
        ValueError: If the phenotypes list is empty or contains non-numeric values.
        TypeError: If phenotypes is not a list.
    """
    if not isinstance(phenotypes, list):
        raise TypeError("Phenotypes input must be a list.")
    if not phenotypes:
        raise ValueError("Phenotypes list cannot be empty.")
    if not all(isinstance(p, (int, float)) for p in phenotypes):
        raise ValueError("All phenotypic values must be numeric.")
    if len(phenotypes) < 2: # Variance requires at least 2 data points
        raise ValueError("Phenotypic variance calculation requires at least two data points.")

    return np.var(phenotypes, ddof=1) # ddof=1 for sample variance


def calculate_h2_from_parent_offspring_regression(
    parent_phenotypes: list[float], offspring_phenotypes: list[float]
) -> float:
    """
    Estimates narrow-sense heritability (h²) using parent-offspring regression.
    h² = 2 * b_op, where b_op is the regression coefficient of offspring on parent.

    Args:
        parent_phenotypes (list[float]): List of parent phenotypic values.
        offspring_phenotypes (list[float]): List of corresponding offspring
                                           phenotypic values.

    Returns:
        float: Estimated narrow-sense heritability (h²).

    Raises:
        ValueError: If lists are empty, not of the same length, or contain
                    non-numeric values.
        TypeError: If inputs are not lists.
    """
    if not isinstance(parent_phenotypes, list) or not isinstance(offspring_phenotypes, list):
        raise TypeError("Parent and offspring phenotypes must be lists.")
    if not parent_phenotypes or not offspring_phenotypes:
        raise ValueError("Phenotype lists cannot be empty.")
    if len(parent_phenotypes) != len(offspring_phenotypes):
        raise ValueError("Parent and offspring phenotype lists must have the same length.")
    if not all(isinstance(p, (int, float)) for p in parent_phenotypes) or \
       not all(isinstance(o, (int, float)) for o in offspring_phenotypes):
        raise ValueError("All phenotypic values must be numeric.")
    if len(parent_phenotypes) < 2: # Regression requires at least 2 data points
        raise ValueError("Regression requires at least two data points for parent and offspring.")


    # Calculate regression coefficient (slope) of offspring on parent
    # polyfit returns [slope, intercept] for degree 1
    b_op, _ = np.polyfit(parent_phenotypes, offspring_phenotypes, 1)

    # h² = 2 * b_op (assuming r_op = 0.5 for parent-offspring)
    h_sq = 2 * b_op
    return h_sq


def partition_variance(
    phenotypic_variance: float, heritability: float
) -> dict[str, float]:
    """
    Partitions total phenotypic variance into genetic and environmental components.

    Args:
        phenotypic_variance (float): Total phenotypic variance (P).
        heritability (float): Narrow-sense heritability (h²).

    Returns:
        dict[str, float]: A dictionary with "genetic_variance" (Va) and
                          "environmental_variance" (Ve).
                          Va = P * h²
                          Ve = P * (1 - h²)

    Raises:
        ValueError: If inputs are non-numeric or heritability is not between 0 and 1.
    """
    if not isinstance(phenotypic_variance, (int, float)) or \
       not isinstance(heritability, (int, float)):
        raise ValueError("Phenotypic variance and heritability must be numeric.")
    if not (0 <= heritability <= 1):
        raise ValueError("Heritability (h²) must be between 0 and 1.")
    if phenotypic_variance < 0:
        raise ValueError("Phenotypic variance cannot be negative.")


    genetic_variance = phenotypic_variance * heritability
    environmental_variance = phenotypic_variance * (1 - heritability)

    return {
        "genetic_variance": genetic_variance,
        "environmental_variance": environmental_variance,
    }
