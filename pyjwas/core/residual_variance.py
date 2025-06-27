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
                     Values are np.ndarray covariance matrices for that pattern.
        """
        self.r0: Optional[np.ndarray] = r0_matrix
        self.ri_dict: Dict[Tuple[bool, ...], np.ndarray] = ri_dict if ri_dict is not None else {}

    def __repr__(self) -> str:
        r0_repr = f"array({self.r0.shape})" if isinstance(self.r0, np.ndarray) else self.r0
        ri_dict_keys = list(self.ri_dict.keys())
        return (f"ResidualVariance(r0={r0_repr}, "
                f"ri_dict_keys={ri_dict_keys})") # Keep repr simple for potentially large dict

    def get_ri_matrix(self, missing_pattern: Tuple[bool, ...]) -> Optional[np.ndarray]:
        """
        Retrieves the residual covariance matrix for a specific missing data pattern.

        Args:
            missing_pattern: A tuple of booleans representing the pattern.

        Returns:
            The covariance matrix for the pattern, or None if not found.
        """
        return self.ri_dict.get(missing_pattern)

    def set_ri_matrix(self, missing_pattern: Tuple[bool, ...], matrix: np.ndarray):
        """
        Sets or updates the residual covariance matrix for a specific missing data pattern.

        Args:
            missing_pattern: A tuple of booleans representing the pattern.
            matrix: The np.ndarray covariance matrix for this pattern.
        """
        self.ri_dict[missing_pattern] = matrix


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
