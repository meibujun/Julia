import scipy.stats
import numpy as np
import math # Moved import math to the top

def single_marker_regression_qtl(genotypes: list[int | float], phenotypes: list[float]) -> tuple[float, float]:
    """
    Performs a single marker regression for QTL mapping.

    This function uses linear regression to test for association between
    genotypes at a single marker locus and quantitative trait phenotypes.
    It returns the F-statistic and corresponding p-value for the regression.

    Args:
        genotypes: A list of numerical genotype codes (e.g., 0 for 'aa',
                   1 for 'Aa', 2 for 'AA'). Can be int or float.
        phenotypes: A list of phenotypic values corresponding to each genotype.

    Returns:
        A tuple (f_statistic, p_value):
            - f_statistic (float): The F-statistic for the regression model.
            - p_value (float): The p-value associated with the F-statistic.
        Returns (0.0, 1.0) if all genotype values are the same (no variation).
        Returns (float('nan'), float('nan')) if degrees of freedom for error is <=0.

    Raises:
        ValueError: If `genotypes` and `phenotypes` lists are not of the same
                    length, if they are empty, or if they have fewer than 3
                    elements (required for F-statistic calculation with df_err > 0).
    """
    if not genotypes or not phenotypes:
        raise ValueError("Input lists (genotypes and phenotypes) cannot be empty.")

    if len(genotypes) != len(phenotypes):
        raise ValueError("Genotype and phenotype lists must have the same length.")

    if len(genotypes) < 3: # n-2 (df_err) must be > 0 for F-statistic
        raise ValueError("At least three data points are required for F-statistic calculation.")

    genotypes_array = np.array(genotypes, dtype=float)
    phenotypes_array = np.array(phenotypes, dtype=float)

    # Check if all genotype values are the same
    if np.unique(genotypes_array).size < 2:
        return 0.0, 1.0 # No variation in predictor, no association possible / F-stat undefined

    # Perform linear regression
    slope, intercept, r_value, p_value_slope, std_err = scipy.stats.linregress(genotypes_array, phenotypes_array)

    # Handle cases where r_value might be nan (e.g. if variance in X or Y is zero, though genotype check helps)
    if np.isnan(r_value):
        # This can happen if all phenotypes are the same, even if genotypes vary.
        # Or if all genotypes are the same (already handled).
        # If phenotypes are all same, slope is 0, r_value is 0 (or nan by some impls).
        # If r_value is nan, r_squared will be nan. F-stat will be nan. p-value for F can be 1.
        return 0.0, 1.0 # No correlation, effectively F=0, p=1

    r_squared = r_value**2
    n = len(genotypes_array)
    df_reg = 1  # Degrees of freedom for regression (number of predictors)
    df_err = n - 2  # Degrees of freedom for error (or residual)

    # df_err <= 0 should have been caught by len(genotypes) < 3, but as a safeguard:
    if df_err <= 0:
        return float('nan'), float('nan') # Should not be reached if initial checks are correct

    # Calculate F-statistic
    # F = (SSR / df_reg) / (SSE / df_err) = (r_squared / df_reg) / ((1 - r_squared) / df_err)
    if math.isclose(1.0 - r_squared, 0.0): # r_squared is very close to 1 (perfect fit)
        if math.isclose(r_squared, 1.0):
            # For perfect correlation, F is theoretically infinite.
            # We can return a very large F and p-value of 0.
            # scipy.stats.f.sf for a very large F will give ~0.
            # Let's use a practical large number if needed, or let scipy handle it.
            # If 1-r_squared is exactly 0, this would be division by zero.
            f_statistic = np.inf
        else: # 1-r_squared is small but not zero, r_squared is not exactly 1. Regular formula applies.
            f_statistic = (r_squared / df_reg) / ((1 - r_squared) / df_err)
    elif math.isclose(r_squared, 0.0): # No correlation
        f_statistic = 0.0
    else: # Standard case
        f_statistic = (r_squared / df_reg) / ((1 - r_squared) / df_err)
        if f_statistic < 0: # Should not happen if r_squared is between 0 and 1
            f_statistic = 0.0


    # Calculate p-value for the F-statistic
    if np.isinf(f_statistic):
        p_value = 0.0
    elif f_statistic == 0.0:
        p_value = 1.0
    else:
        p_value = scipy.stats.f.sf(f_statistic, df_reg, df_err)

    return f_statistic, p_value
