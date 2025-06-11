import numpy as np
from animal_breeding_genetics.matrix_ops import invert_matrix, multiply_matrices

def calculate_selection_index_b_values(
    P_matrix: np.ndarray, G_vector: np.ndarray
) -> np.ndarray:
    """
    Calculates selection index weights (b-values).

    Formula: b = P_inverse @ G

    Args:
        P_matrix (np.ndarray): Phenotypic variance-covariance matrix of the
                               information sources (n x n).
        G_vector (np.ndarray): Vector of genetic covariances between the
                               information sources and the aggregate genotype
                               (or breeding value for a single trait) (n x 1).

    Returns:
        np.ndarray: Vector of selection index weights (b-values) (n x 1).

    Raises:
        ValueError: If P_matrix is not square, or if dimensions are incompatible.
        np.linalg.LinAlgError: If P_matrix is singular.
    """
    if P_matrix.ndim != 2 or P_matrix.shape[0] != P_matrix.shape[1]:
        raise ValueError("P_matrix must be a square matrix.")
    if G_vector.ndim == 1: # Ensure G_vector is a column vector
        G_vector = G_vector.reshape(-1,1)
    if G_vector.ndim != 2 or G_vector.shape[1] != 1:
        raise ValueError("G_vector must be a column vector (n x 1).")

    if P_matrix.shape[0] != G_vector.shape[0]:
        raise ValueError(
            "P_matrix and G_vector must have compatible dimensions "
            f"(P: {P_matrix.shape}, G: {G_vector.shape})."
        )

    P_inv_matrix = invert_matrix(P_matrix)
    b_vector = multiply_matrices(P_inv_matrix, G_vector)
    return b_vector


def calculate_selection_index_accuracy(
    b_vector: np.ndarray,
    G_vector: np.ndarray,
    var_aggregate_genotype: float,
) -> float:
    """
    Calculates the accuracy of a selection index.

    Formula: r_IH = sqrt((b_transpose @ G_vector) / var_aggregate_genotype)
    This formula assumes b_vector is optimal (derived from P_inv @ G).

    Args:
        b_vector (np.ndarray): Selection index weights (n x 1 or 1D).
        G_vector (np.ndarray): Genetic covariance vector between information
                               sources and aggregate genotype (n x 1 or 1D).
        var_aggregate_genotype (float): Variance of the aggregate genotype (scalar).
                                        Must be non-negative.

    Returns:
        float: Accuracy of the selection index (correlation between index (I)
               and aggregate genotype (H)).

    Raises:
        ValueError: If dimensions are incompatible, var_aggregate_genotype is negative,
                    or the value under the square root is negative (beyond tolerance).
    """
    if b_vector.ndim == 1:
        b_vector = b_vector.reshape(-1,1)
    if G_vector.ndim == 1:
        G_vector = G_vector.reshape(-1,1)

    if b_vector.shape[1] != 1:
        raise ValueError("b_vector must be effectively a column vector (n x 1).")
    if G_vector.shape[1] != 1:
        raise ValueError("G_vector must be effectively a column vector (n x 1).")
    if b_vector.shape[0] != G_vector.shape[0]:
        raise ValueError(
            "b_vector and G_vector must have the same number of rows."
        )
    if not isinstance(var_aggregate_genotype, (int, float)):
        raise TypeError("var_aggregate_genotype must be a numeric value.")
    if var_aggregate_genotype < 0:
        raise ValueError("var_aggregate_genotype must be non-negative.")
    if var_aggregate_genotype == 0: # If H has no variance, accuracy is undefined or 0.
        return 0.0                 # Let's define as 0 if there's nothing to predict.

    b_transpose_G = multiply_matrices(b_vector.T, G_vector)
    numerator = b_transpose_G[0, 0]

    # Numerator (Cov(I,H)) can be slightly negative due to float precision with optimal b
    # but should ideally be non-negative.
    if numerator < -1e-9:
        raise ValueError(
            f"Numerator (b_transpose @ G_vector = {numerator:.4e}) for accuracy calculation is negative beyond tolerance. "
            "This may indicate sub-optimal b-values or issues with P and G matrices."
            )
    if numerator < 0: numerator = 0.0 # Correct small float precision issues

    value_under_sqrt = numerator / var_aggregate_genotype

    if value_under_sqrt > 1.0 + 1e-9: # Allow small tolerance for value slightly > 1
        # This warning can be useful for debugging consistency of P, G, var_H
        # print(f"Warning: Accuracy calculation value_under_sqrt = {value_under_sqrt:.6f} > 1. Capping to 1.")
        value_under_sqrt = 1.0
    if value_under_sqrt < 0: # Should be caught by numerator check if var_agg_genotype is positive
         value_under_sqrt = 0.0

    accuracy = np.sqrt(value_under_sqrt)
    return accuracy


def index_own_performance(h_squared: float, phenotypic_sd: float) -> tuple[np.ndarray, float]:
    """
    Calculates selection index weight and accuracy for an index based on an
    individual's own performance for a single trait.

    Args:
        h_squared (float): Narrow-sense heritability of the trait.
        phenotypic_sd (float): Phenotypic standard deviation of the trait.

    Returns:
        tuple[np.ndarray, float]:
            - b_value (np.ndarray): The selection index weight (scalar, as 1x1 array).
            - accuracy (float): The accuracy of the selection index.
    """
    if not (-1e-9 <= h_squared <= 1.0 + 1e-9):
        raise ValueError("Heritability (h_squared) must be between 0 and 1.")
    h_squared = np.clip(h_squared, 0, 1)
    if phenotypic_sd < 0:
        raise ValueError("Phenotypic standard deviation must be non-negative.")

    phenotypic_variance = phenotypic_sd**2
    genetic_variance = h_squared * phenotypic_variance

    if phenotypic_variance == 0:
        return np.array([[0.0]]), 0.0
    if genetic_variance == 0: # This also covers h_squared = 0
        return np.array([[0.0]]), 0.0

    P_matrix = np.array([[phenotypic_variance]])
    G_vector = np.array([[genetic_variance]])
    var_aggregate_genotype = genetic_variance

    b_value_vector = calculate_selection_index_b_values(P_matrix, G_vector)
    accuracy = calculate_selection_index_accuracy(
        b_value_vector, G_vector, var_aggregate_genotype
    )
    return b_value_vector, accuracy


def index_progeny_records(
    n_progeny: int, h_squared: float, phenotypic_sd: float, relationship_sire_progeny: float = 0.5
) -> tuple[np.ndarray, float]:
    """
    Calculates selection index weight and accuracy for sire evaluation based on
    the mean performance of n progeny for a single trait.

    Args:
        n_progeny (int): Number of progeny records. Must be positive.
        h_squared (float): Narrow-sense heritability of the trait.
        phenotypic_sd (float): Phenotypic standard deviation of the trait.
        relationship_sire_progeny (float, optional): Coefficient of relationship
            between the sire and each progeny. Defaults to 0.5.

    Returns:
        tuple[np.ndarray, float]:
            - b_value (np.ndarray): The selection index weight (scalar, as 1x1 array).
            - accuracy (float): The accuracy of the selection index.
    """
    if n_progeny <= 0:
        raise ValueError("Number of progeny (n_progeny) must be positive.")
    if not (-1e-9 <= h_squared <= 1.0 + 1e-9):
        raise ValueError("Heritability (h_squared) must be between 0 and 1.")
    h_squared = np.clip(h_squared, 0, 1)
    if phenotypic_sd < 0:
        raise ValueError("Phenotypic standard deviation must be non-negative.")
    if not (0 < relationship_sire_progeny <= 1):
        raise ValueError("Relationship sire-progeny must be > 0 and <= 1.")

    phenotypic_variance = phenotypic_sd**2
    genetic_variance = h_squared * phenotypic_variance

    if phenotypic_variance == 0: return np.array([[0.0]]), 0.0
    if genetic_variance == 0: return np.array([[0.0]]), 0.0

    t_hs_pheno = 0.25 * h_squared # Phenotypic correlation among half-sibs (progeny)
    var_progeny_mean = (phenotypic_variance / n_progeny) * (1 + (n_progeny - 1) * t_hs_pheno)

    if var_progeny_mean <= 1e-9: # Effectively zero variance for the information source
        return np.array([[0.0]]), 0.0

    P_matrix = np.array([[var_progeny_mean]])
    cov_sire_bv_progeny_phenotype = relationship_sire_progeny * genetic_variance
    G_vector = np.array([[cov_sire_bv_progeny_phenotype]])
    var_aggregate_genotype = genetic_variance

    b_value_vector = calculate_selection_index_b_values(P_matrix, G_vector)
    accuracy = calculate_selection_index_accuracy(
        b_value_vector, G_vector, var_aggregate_genotype
    )
    return b_value_vector, accuracy


def index_own_and_full_sib_mean(
    h_squared: float, phenotypic_sd: float, n_sibs: int, c_squared: float = 0.0
) -> tuple[np.ndarray, float]:
    """
    Calculates index weights and accuracy for an index combining own performance
    and the mean performance of n full-sibs.

    Args:
        h_squared (float): Narrow-sense heritability.
        phenotypic_sd (float): Phenotypic standard deviation.
        n_sibs (int): Number of full-sibs in the mean. Must be positive.
        c_squared (float, optional): Proportion of phenotypic variance due to
                                     common environmental effects for full-sibs. Defaults to 0.0.

    Returns:
        tuple[np.ndarray, float]:
            - b_values (np.ndarray): Selection index weights (2x1 vector).
            - accuracy (float): Accuracy of the index.
    """
    if not (-1e-9 <= h_squared <= 1.0 + 1e-9):
        raise ValueError("Heritability (h_squared) must be between 0 and 1.")
    h_squared = np.clip(h_squared, 0, 1)
    if phenotypic_sd < 0:
        raise ValueError("Phenotypic standard deviation must be non-negative.")
    if n_sibs <= 0:
        raise ValueError("Number of sibs (n_sibs) must be positive.")
    if not (-1e-9 <= c_squared < 1.0 + 1e-9):
        raise ValueError("Common environmental variance (c_squared) must be between 0 and 1.")
    c_squared = np.clip(c_squared, 0, 1)
    if h_squared + c_squared > 1.0 + 1e-9 :
        raise ValueError("Sum of h_squared and c_squared cannot exceed 1.")

    PV = phenotypic_sd**2
    GV = h_squared * PV
    Var_Ec = c_squared * PV

    if PV == 0: return np.zeros((2,1)), 0.0
    if GV == 0: # If no genetic variance, G vector is all zeros, b is all zeros, accuracy is 0
        return np.zeros((2,1)), 0.0

    P11 = PV
    t_fs_pheno = 0.5 * h_squared + c_squared
    P22 = (PV / n_sibs) * (1 + (n_sibs - 1) * t_fs_pheno)
    P12 = 0.5 * GV + Var_Ec # Cov(own_pheno, one_sib_pheno)

    P_matrix = np.array([[P11, P12], [P12, P22]])

    G1 = GV
    G2 = 0.5 * GV
    G_vector = np.array([[G1], [G2]])
    var_aggregate_genotype = GV

    try:
        b_values = calculate_selection_index_b_values(P_matrix, G_vector)
        accuracy = calculate_selection_index_accuracy(b_values, G_vector, var_aggregate_genotype)
    except np.linalg.LinAlgError:
        # This can happen if P is singular (e.g. if X1 and X2 are perfectly correlated)
        # Example: n_sibs=1 and own record is the "sib record" and c_squared + 0.5*h_squared = 1
        # If P is singular, but G is not "in the image" of P, b is problematic.
        # If GV=0, already handled. If P is singular due to perfect correlation,
        # one info source is redundant. A robust way is to use pseudo-inverse or simplify.
        # For now, let's assume if P is singular, the index is ill-defined this way.
        # However, if G is in the column space of P, a solution might exist.
        # A simple case: if P11=P12=P22=PV (perfect correlation), P_inv fails.
        # But if G1=G2=GV, then b might be [1,0] or [0,1] with accuracy sqrt(h2).
        # This is too complex for this function; better to ensure P is well-conditioned.
        raise # Re-raise for now

    return b_values, accuracy


def economic_selection_index(
    economic_weights: np.ndarray,
    P_matrix_info: np.ndarray,
    G_matrix_traits_info: np.ndarray,
    V_A_matrix_traits: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """
    Calculates an economic selection index.

    Args:
        economic_weights (np.ndarray): Vector of economic weights for each trait
                                       in the aggregate genotype (m x 1 or 1D).
        P_matrix_info (np.ndarray): Phenotypic var-cov matrix of information
                                    sources (n x n).
        G_matrix_traits_info (np.ndarray): Matrix of genetic covariances between
                                           traits in agg. genotype (rows, m) and
                                           info. sources (columns, n). (m x n)
        V_A_matrix_traits (np.ndarray): Genetic var-cov matrix of the traits
                                        in the aggregate genotype (m x m).

    Returns:
        tuple[np.ndarray, float, float]:
            - b_vector (np.ndarray): Index weights (n x 1).
            - accuracy_index (float): Accuracy of the index (r_IH).
            - var_H (float): Variance of the aggregate genotype (H).
    """
    if economic_weights.ndim == 1:
        economic_weights = economic_weights.reshape(-1,1)
    if economic_weights.shape[1] != 1:
        raise ValueError("economic_weights must be a column vector (m x 1).")

    m = economic_weights.shape[0] # Number of traits in aggregate genotype
    n = P_matrix_info.shape[0]  # Number of information sources

    if P_matrix_info.ndim != 2 or P_matrix_info.shape[0] != P_matrix_info.shape[1]:
        raise ValueError("P_matrix_info must be a square matrix (n x n).")
    if G_matrix_traits_info.ndim != 2 or G_matrix_traits_info.shape[0] != m or G_matrix_traits_info.shape[1] != n:
        raise ValueError("G_matrix_traits_info must have dimensions (m x n).")
    if V_A_matrix_traits.ndim != 2 or V_A_matrix_traits.shape[0] != m or V_A_matrix_traits.shape[1] != m:
        raise ValueError("V_A_matrix_traits must be a square matrix (m x m).")

    # Aggregate Genotype H = economic_weights.T @ g_traits
    # var_H = economic_weights.T @ V_A_matrix_traits @ economic_weights
    var_H_matrix = multiply_matrices(economic_weights.T, V_A_matrix_traits)
    var_H = multiply_matrices(var_H_matrix, economic_weights)[0,0]

    if var_H < 0:
        raise ValueError(f"Variance of aggregate genotype (var_H = {var_H:.4e}) is negative.")
    if var_H == 0: # No variance in the breeding objective
        return np.zeros((n,1)), 0.0, 0.0

    # Covariance vector between Information sources (I) and Aggregate Genotype (H)
    # C_IH = Cov(I, H) = Cov(I, economic_weights.T @ g_traits)
    #      = Cov(I, sum(a_j * g_trait_j)) = sum(a_j * Cov(I, g_trait_j))
    # Cov(I, g_trait_j) is the j-th column of G_matrix_traits_info.T
    # So C_IH = G_matrix_traits_info.T @ economic_weights (n x 1)
    C_IH_vector = multiply_matrices(G_matrix_traits_info.T, economic_weights)

    # Index weights b = P_matrix_info_inverse @ C_IH_vector
    b_vector = calculate_selection_index_b_values(P_matrix_info, C_IH_vector)

    # Accuracy r_IH = sqrt((b.T @ C_IH_vector) / var_H)
    accuracy = calculate_selection_index_accuracy(b_vector, C_IH_vector, var_H)

    return b_vector, accuracy, var_H
