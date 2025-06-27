from typing import List, Any, Optional, Union
import numpy as np
from scipy.sparse import spmatrix # For type hinting sparse matrix

class ModelTerm:
    """
    Represents a single term in a model equation (e.g., "y1:A", "y2:A*B").
    Translated from the Julia JWAS.jl ModelTerm struct.
    """
    def __init__(self, term_str: str, model_index: int, trait_name: str):
        """
        Initializes a ModelTerm instance.

        Args:
            term_str (str): The term string, e.g., "A", "A*B".
            model_index (int): The index of the model equation this term belongs to.
            trait_name (str): The name of the trait this term is associated with.
        """
        _term_str_stripped: str = term_str.strip()
        _trait_name_stripped: str = trait_name.strip()

        factor_vec: List[str] = [f.strip() for f in _term_str_stripped.split('*')]

        self.imodel: int = model_index
        self.itrait: str = _trait_name_stripped
        # The fully qualified term string, e.g., "y1:A" or "y1:A*B"
        self.trm_str: str = f"{_trait_name_stripped}:{_term_str_stripped}"

        self.n_factors: int = len(factor_vec)
        # In Python, symbols are typically represented as strings
        self.factors: List[str] = factor_vec

        # These fields are populated later during data processing and MME construction
        self.data: List[str] = []  # String representation of levels/values for each observation
        self.val: Optional[np.ndarray] = None # Numerical values for each observation (Float32 or Float64)

        self.n_levels: int = 0 # Number of unique levels for this term
        self.names: List[Any] = [] # Names of the levels

        self.start_pos: int = 0 # Starting column position in the combined incidence matrix
        self.X: Optional[Union[np.ndarray, spmatrix]] = None # Incidence matrix for this term (can be sparse)

        self.random_type: str = "fixed" # Default type, can be changed (e.g., "random", "genotypes")

    def __repr__(self) -> str:
        return (f"ModelTerm(trm_str='{self.trm_str}', imodel={self.imodel}, itrait='{self.itrait}', "
                f"n_factors={self.n_factors}, factors={self.factors}, random_type='{self.random_type}', "
                f"n_levels={self.n_levels}, start_pos={self.start_pos})")

    # Potential future methods:
    # - Method to set data and values
    # - Method to build/set incidence matrix X
    # - Properties for read-only attributes if needed.

if __name__ == '__main__':
    # Example Usage:
    term1 = ModelTerm("age", model_index=1, trait_name="y1")
    print(term1)
    term1.n_levels = 1
    term1.random_type = "covariate" # Example modification
    print(term1)

    term2 = ModelTerm("herd * year", model_index=1, trait_name="y1")
    print(term2)
    term2.factors = ["herd", "year"] # Example, constructor does this
    print(term2)

    # To simulate population during MME building:
    # term1.X = np.array([[1],[2],[3]])
    # term1.val = np.array([10,20,30], dtype=np.float32)
    # term1.names = ["age"]
    # term1.n_levels = 1
    # print(f"Term 1 X: {term1.X}, Val: {term1.val}, Names: {term1.names}")
