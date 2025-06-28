from typing import Dict, Tuple, Optional
import numpy as np

class ResidualVariance:
    """
    Holds residual covariance matrices, particularly for handling missing data patterns
    in multi-trait models.
    Translated from the Julia JWAS.jl ResVar struct.
    """
    def __init__(self,
                 r0_matrix: Optional[np.ndarray] = None,
                 ri_dict: Optional[Dict[Tuple[bool, ...], np.ndarray]] = None):
        """
        Initializes a ResidualVariance instance.

        Args:
            r0_matrix: The base residual covariance matrix (e.g., for complete data).
                       np.ndarray of shape (n_traits, n_traits).
            ri_dict: A dictionary mapping missing data patterns to their
                     corresponding residual covariance matrices.
                     Keys are tuples of booleans (e.g., (True, False, True) means
                     trait 1 observed, trait 2 missing, trait 3 observed).
                     Values are np.ndarray representing the sub-block of R_inv for the observed traits,
                     embedded in a full n_traits x n_traits matrix.
        """
        self.r0_full_R_matrix: Optional[np.ndarray] = r0_matrix # Store the full R matrix used to generate this dict
        self.ri_pattern_inverses: Dict[Tuple[bool, ...], np.ndarray] = ri_dict if ri_dict is not None else {}

    def __repr__(self) -> str:
        r0_repr = f"array({self.r0_full_R_matrix.shape})" if isinstance(self.r0_full_R_matrix, np.ndarray) else self.r0_full_R_matrix
        ri_dict_keys = list(self.ri_pattern_inverses.keys()) # Corrected here
        return (f"ResidualVariance(r0_full_R_matrix_shape={r0_repr}, " # Clarified repr
                f"num_cached_patterns={len(ri_dict_keys)})")

    def get_or_compute_R_inv_for_pattern(self,
                                         pattern: Tuple[bool, ...],
                                         full_R_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieves or computes the inverse of the residual covariance sub-matrix
        for a specific missing data pattern. The result is embedded in a full-size matrix.

        Args:
            pattern: Tuple of booleans indicating observed (True) / missing (False) traits.
            full_R_matrix: The current full (n_traits x n_traits) residual covariance matrix.
                           If None, uses self.r0_full_R_matrix. This matrix (R0) should be
                           updated if the main R estimate changes in MCMC.

        Returns:
            An (n_traits x n_traits) matrix where the block corresponding to observed
            traits contains the inverse of that sub-block of R, and other elements are zero.
        """
        if pattern in self.ri_pattern_inverses:
            # TODO: Add a check if full_R_matrix has changed since this was cached.
            # For now, assume if cached, it's valid for the R0 it was built with.
            # If self.r0_full_R_matrix is not None and full_R_matrix is not None and
            # not np.array_equal(self.r0_full_R_matrix, full_R_matrix):
            # print("Warning: full_R_matrix changed, but returning cached R_inv_for_pattern.")
            return self.ri_pattern_inverses[pattern]

        current_R_matrix = full_R_matrix if full_R_matrix is not None else self.r0_full_R_matrix
        if current_R_matrix is None:
            raise ValueError("Full R matrix (r0_full_R_matrix or argument) must be provided to compute pattern inverse.")

        n_traits = current_R_matrix.shape[0]
        if len(pattern) != n_traits:
            raise ValueError(f"Pattern length {len(pattern)} does not match R matrix dimension {n_traits}.")

        observed_indices = np.where(pattern)[0]
        if len(observed_indices) == 0: # All traits missing for this pattern
            # Return a zero matrix of appropriate size, as there's no information.
            RZ = np.zeros((n_traits, n_traits))
            self.ri_pattern_inverses[pattern] = RZ
            return RZ

        R_sub_observed = current_R_matrix[np.ix_(observed_indices, observed_indices)]

        try:
            R_sub_observed_inv = np.linalg.inv(R_sub_observed)
        except np.linalg.LinAlgError:
            # Sub-matrix is singular, cannot invert. This implies issues with R or pattern.
            # Fallback: use pseudo-inverse or return matrix of zeros/large values?
            # For now, error or return zeros. JWAS might handle this differently.
            # Let's return a zero matrix for this pattern's contribution.
            print(f"Warning: Sub-matrix of R for pattern {pattern} is singular. Using zero block.")
            RZ = np.zeros((n_traits, n_traits))
            self.ri_pattern_inverses[pattern] = RZ
            return RZ

        RZ = np.zeros((n_traits, n_traits))
        for i, r_idx in enumerate(observed_indices):
            for j, c_idx in enumerate(observed_indices):
                RZ[r_idx, c_idx] = R_sub_observed_inv[i, j]

        self.ri_pattern_inverses[pattern] = RZ
        # Update the r0_full_R_matrix if this is the first computation or if full_R_matrix was new
        if self.r0_full_R_matrix is None or (full_R_matrix is not None and not np.array_equal(self.r0_full_R_matrix, full_R_matrix)):
            self.r0_full_R_matrix = np.copy(full_R_matrix)

        return RZ

    # Keep old methods for compatibility or remove if get_or_compute replaces them fully
    def get_ri_matrix(self, missing_pattern: Tuple[bool, ...]) -> Optional[np.ndarray]:
        return self.ri_pattern_inverses.get(missing_pattern)

    def set_ri_matrix(self, missing_pattern: Tuple[bool, ...], matrix: np.ndarray):
        self.ri_pattern_inverses[missing_pattern] = matrix


if __name__ == '__main__':
    # Example Usage:
    # Base residual covariance for 2 traits
    r0 = np.array([[10.0, 2.0], [2.0, 5.0]])

    res_var_handler = ResidualVariance(r0_matrix=r0)
    print(res_var_handler)

    # Pattern: trait 1 observed, trait 2 missing
    pattern1 = (True, False)
    # For this pattern, only variance of trait 1 is relevant, or a modified matrix
    r_pattern1 = np.array([[10.0]]) # Simplified for example
    res_var_handler.set_ri_matrix(pattern1, r_pattern1)
    print(res_var_handler)
    print(f"Matrix for pattern {pattern1}: {res_var_handler.get_ri_matrix(pattern1)}")

    # Pattern: trait 1 missing, trait 2 observed
    pattern2 = (False, True)
    r_pattern2 = np.array([[5.0]]) # Simplified
    res_var_handler.set_ri_matrix(pattern2, r_pattern2)
    print(res_var_handler)

    # Pattern: both traits observed (might be same as r0 or specific if r0 is for full block)
    pattern_both = (True, True)
    res_var_handler.set_ri_matrix(pattern_both, r0) # Using r0 for this example
    print(f"Matrix for pattern {pattern_both}: {res_var_handler.get_ri_matrix(pattern_both)}")
