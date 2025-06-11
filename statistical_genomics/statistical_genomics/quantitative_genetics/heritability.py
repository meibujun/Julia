import scipy.stats
import numpy as np

def estimate_narrow_sense_heritability_po_regression(
    parent_mid_values: list[float],
    offspring_values: list[float]
) -> float:
    """
    Estimates narrow-sense heritability (h^2) using parent-offspring regression.

    The slope of the linear regression of offspring phenotypic values on the
    mid-parent phenotypic values is an estimator of h^2.

    Args:
        parent_mid_values: A list of phenotypic values representing the mid-point
                           value for each pair of parents.
        offspring_values: A list of phenotypic values for the offspring, corresponding
                          to each mid-parent value in `parent_mid_values`.

    Returns:
        The estimated narrow-sense heritability (h^2), which is the slope
        of the parent-offspring regression line.

    Raises:
        ValueError: If `parent_mid_values` and `offspring_values` are not of the
                    same length, if they are empty, or if they have fewer than 2
                    elements (required for meaningful regression).
    """
    if not parent_mid_values or not offspring_values:
        raise ValueError("Input lists (parent_mid_values and offspring_values) cannot be empty.")

    if len(parent_mid_values) != len(offspring_values):
        raise ValueError("Parent and offspring value lists must have the same length.")

    if len(parent_mid_values) < 2:
        raise ValueError("At least two data points are required for regression analysis.")

    # Convert lists to numpy arrays for scipy.stats.linregress
    # While linregress can often handle lists, using np.array is safer and common practice.
    parent_array = np.array(parent_mid_values, dtype=float)
    offspring_array = np.array(offspring_values, dtype=float)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(parent_array, offspring_array)

    # Narrow-sense heritability (h^2) is estimated by the slope of this regression
    h_squared_estimate = slope

    return h_squared_estimate

# Example usage:
# parents = [1.0, 2.0, 3.0, 4.0, 5.0]
# offspring_h1 = [1.0, 2.0, 3.0, 4.0, 5.0] # h^2 = 1.0
# offspring_h05 = [0.5, 1.0, 1.5, 2.0, 2.5] # h^2 = 0.5
# offspring_h0 = [3.0, 3.0, 3.0, 3.0, 3.0] # h^2 = 0.0
#
# print(f"h^2 (slope) for perfect heritability: {estimate_narrow_sense_heritability_po_regression(parents, offspring_h1)}")
# print(f"h^2 (slope) for h^2=0.5: {estimate_narrow_sense_heritability_po_regression(parents, offspring_h05)}")
# print(f"h^2 (slope) for h^2=0: {estimate_narrow_sense_heritability_po_regression(parents, offspring_h0)}")
