import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, diags, eye as sparse_eye
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Optional

from ..pedigree import PedigreeData, calculate_A_inverse
from ..core.genotypes import GenotypesData
from ..genotypes_io.utils_geno import make_incidence_matrix_for_ids

def _calculate_GRM_vanraden1( # Keep as internal helper if only used here
    genotype_matrix: np.ndarray,
    allele_freqs_p: Optional[np.ndarray] = None,
    center_before_std: bool = True
) -> np.ndarray:
    M = genotype_matrix
    n_obs, n_markers = M.shape

    current_allele_freqs_p = allele_freqs_p
    if current_allele_freqs_p is None:
        marker_means = np.mean(M, axis=0)
        current_allele_freqs_p = marker_means / 2.0

    if current_allele_freqs_p.shape[0] != n_markers:
        raise ValueError("Allele frequencies length mismatch with number of markers.")

    denom_scaling_factor_per_marker = 2 * current_allele_freqs_p * (1 - current_allele_freqs_p)

    Z_for_grm = M
    if center_before_std:
        mat_2P = 2 * current_allele_freqs_p
        Z_for_grm = M - mat_2P[np.newaxis, :]

    sum_2pq = np.sum(denom_scaling_factor_per_marker)
    if sum_2pq < 1e-8:
        raise ValueError("Sum of 2pq is zero or near zero for GRM calculation.")

    G = (Z_for_grm @ Z_for_grm.T) / sum_2pq
    return G


def calculate_H_inverse(
    pedigree_data: PedigreeData,
    geno_data_for_grm: GenotypesData,
    weight_G: float = 0.95
) -> Tuple[csc_matrix, List[str]]:
    """
    Calculates the H-inverse matrix for single-step genomic evaluations.
    H_inv = A_inv + [ 0             0          ]
                    [ 0   inv(G_tuned) - A22_inv_direct ]
    where G_tuned = weight_G * G_aligned + (1-weight_G) * A22_from_A.
    A22_inv_direct is calculated from A_inv partitions for stability and efficiency.
    A22_from_A (for blending) is calculated via full A matrix inversion (matches JWAS.jl).

    Args:
        pedigree_data: PedigreeData object, F coefficients must be calculated.
        geno_data_for_grm: GenotypesData where .genotypes is the GRM (G)
                           and .obs_ids are the IDs for G. Must have .is_grm = True.
        weight_G: Weight for G when blending G with A22.

    Returns:
        A tuple: (H_inverse_matrix, ordered_ids_for_H).
    """
    if not geno_data_for_grm.is_grm or geno_data_for_grm.genotypes is None:
        raise ValueError("geno_data_for_grm must contain a GRM.")
    if not (0.0 <= weight_G <= 1.0):
        print("Warning: weight_G clamped to [0,1].")
        weight_G = max(0.0, min(1.0, weight_G))

    genotyped_animal_ids_from_grm = geno_data_for_grm.obs_ids
    G_original = geno_data_for_grm.genotypes

    n_non_genotyped = pedigree_data.set_genotyped_animals(genotyped_animal_ids_from_grm)
    ordered_ids_for_H = pedigree_data.get_ordered_ids_str()
    n_total_ped = len(ordered_ids_for_H)
    n_genotyped = len(genotyped_animal_ids_from_grm)

    if G_original.shape != (n_genotyped, n_genotyped):
        raise ValueError("Original G_matrix shape mismatch.")

    print("Calculating A-inverse for reordered pedigree...")
    A_inv = calculate_A_inverse(pedigree_data) # Sparse CSC matrix

    # --- Calculate A22_inv directly from A_inv partitions (more stable) ---
    print("Calculating A22_inv from A_inv partitions...")
    idx_n = slice(0, n_non_genotyped)
    idx_g = slice(n_non_genotyped, n_total_ped)

    A_inv_gg = A_inv[idx_g, idx_g]
    A22_inv_direct: Optional[csc_matrix] = None

    if n_non_genotyped > 0 :
        A_inv_nn = A_inv[idx_n, idx_n]
        A_inv_ng = A_inv[idx_n, idx_g]
        A_inv_gn = A_inv[idx_g, idx_n]

        try:
            # Ensure A_inv_nn is CSC for spsolve
            A_inv_nn_csc = A_inv_nn.tocsc() if not isinstance(A_inv_nn, csc_matrix) else A_inv_nn

            # Solve A_inv_nn @ X = A_inv_ng for X. X = inv(A_inv_nn) @ A_inv_ng
            # spsolve expects B to be dense vector or matrix for some solvers.
            A_inv_ng_dense = A_inv_ng.toarray() if isinstance(A_inv_ng, spmatrix) else A_inv_ng

            # Check if A_inv_nn_csc is singular before solving by trying to factorize
            try:
                _ = spsolve(A_inv_nn_csc, np.arange(A_inv_nn_csc.shape[0])) # Test solve
            except Exception as factor_error: # More specific errors could be RuntimeError for UMFPACK/SuperLU
                 print(f"Warning: A_inv_nn seems singular or problem with sparse solve ({factor_error}). Using dense pseudo-inverse for inv(A_inv_nn).")
                 inv_A_inv_nn = np.linalg.pinv(A_inv_nn.toarray())
                 A_inv_nn_dot_A_inv_ng_result = inv_A_inv_nn @ A_inv_ng_dense
            else:
                 A_inv_nn_dot_A_inv_ng_result = spsolve(A_inv_nn_csc, A_inv_ng_dense)

            term_to_subtract = A_inv_gn @ A_inv_nn_dot_A_inv_ng_result
            A22_inv_direct = A_inv_gg - term_to_subtract
            if not isinstance(A22_inv_direct, csc_matrix): A22_inv_direct = csc_matrix(A22_inv_direct)
        except Exception as e:
            print(f"Warning: Could not compute A22_inv using A_inv partitions: {e}. H_inv might be inaccurate if A22 from full A is also problematic.")
            A22_inv_direct = None # Fallback will be inv(A22_from_A_full)
    else: # All animals genotyped
        A22_inv_direct = A_inv_gg # Which is A_inv itself

    # --- Calculate A22 (for blending G) by inverting full A_inv ---
    # This matches JWAS.jl's approach for getting A22 for blending.
    A22_for_blending: Optional[np.ndarray] = None
    print("Calculating A_full to extract A22 for G-A22 blending (can be slow)...")
    try:
        if A_inv.shape[0] == 0: raise ValueError("A_inv is empty for A_full calculation.")
        A_full = np.linalg.inv(A_inv.toarray())
        A22_for_blending = A_full[n_non_genotyped:, n_non_genotyped:]
    except np.linalg.LinAlgError:
        print("Warning: Full A_inv is singular when calculating A_full for A22. Using pseudo-inverse.")
        try:
            A_full = np.linalg.pinv(A_inv.toarray())
            A22_for_blending = A_full[n_non_genotyped:, n_non_genotyped:]
        except Exception as e_pinv:
            print(f"Error using pseudo-inverse for A_full to get A22: {e_pinv}. Cannot proceed with G-A22 blending.")
            raise ValueError("Cannot obtain A22 for blending G.")

    # --- Align G and Tune ---
    genotyped_ids_in_H_order = ordered_ids_for_H[n_non_genotyped:]
    G_aligned = G_original
    if list(genotyped_animal_ids_from_grm) != list(genotyped_ids_in_H_order):
        print("Aligning G matrix with H matrix order for genotyped animals...")
        Z_align_G = make_incidence_matrix_for_ids(genotyped_ids_in_H_order, genotyped_animal_ids_from_grm)
        G_aligned = Z_align_G @ G_original @ Z_align_G.T
        if not isinstance(G_aligned, np.ndarray): G_aligned = G_aligned.toarray()

    print("Tuning G matrix...")
    G_tuned = weight_G * G_aligned + (1.0 - weight_G) * A22_for_blending
    try:
        np.linalg.cholesky(G_tuned)
    except np.linalg.LinAlgError:
        print("Warning: Tuned G matrix not PD. Adding epsilon to diagonal.")
        G_tuned += np.eye(G_tuned.shape[0]) * 1e-6
        try: np.linalg.cholesky(G_tuned)
        except np.linalg.LinAlgError: raise ValueError("Tuned G matrix not PD after adding epsilon.")

    # --- Inverses for H_inv formula ---
    print("Calculating inv(G_tuned)...")
    try: G_tuned_inv = np.linalg.inv(G_tuned)
    except np.linalg.LinAlgError: raise ValueError("Tuned G matrix is singular.")

    A22_inv_for_formula = A22_inv_direct
    if A22_inv_for_formula is None: # If partition method failed
        print("Using A22_inv by inverting A22 (from A_full) for H_inv formula...")
        try: A22_inv_for_formula = np.linalg.inv(A22_for_blending)
        except np.linalg.LinAlgError:
            print("Warning: A22 matrix (from A_full) is singular. Using pseudo-inverse for A22_inv in H_inv.")
            A22_inv_for_formula = np.linalg.pinv(A22_for_blending)

    # --- Construct H_inv ---
    print("Constructing H-inverse...")
    H_inv_lil = A_inv.tolil() if isinstance(A_inv, (csc_matrix, csr_matrix)) else lil_matrix(A_inv)

    block_to_add = G_tuned_inv - (A22_inv_for_formula.toarray() if isinstance(A22_inv_for_formula, spmatrix) else A22_inv_for_formula)

    if H_inv_lil[idx_g, idx_g].shape == block_to_add.shape:
        H_inv_lil[idx_g, idx_g] += block_to_add
    else:
        raise ValueError("Shape mismatch adding G_inv - A22_inv block to H_inv.")

    print("H-inverse construction complete.")
    return H_inv_lil.tocsc(), ordered_ids_for_H


if __name__ == '__main__':
    # ... (rest of __main__ block remains the same) ...
    print("--- Testing Single-Step Relationship Utilities ---")
    dummy_ped_ss_path = "dummy_ped_ss.csv" # Define for cleanup if test fails early
    try:
        ped_content_ss = """
        1,0,0
        2,0,0
        3,1,2
        4,1,0
        5,3,4
        """
        with open(dummy_ped_ss_path, "w") as f: f.write(ped_content_ss.strip())
        ped = read_pedigree(dummy_ped_ss_path)

        genotyped_ids_list = ["3", "4", "5"]

        # G must be ordered according to geno_data_for_grm.obs_ids
        # calculate_H_inverse will align it to the H matrix order of genotyped animals.
        # Let's set G's original obs_ids to ["3", "4", "5"] for simplicity matching Aguilar example order.
        grm_obs_ids_orig = ["3", "4", "5"]
        G_original_for_345 = np.array([[1.1, 0.1, 0.2],
                                     [0.1, 1.0, 0.15],
                                     [0.2, 0.15, 1.05]])
        G_original_for_345 = (G_original_for_345 + G_original_for_345.T) / 2.0
        G_original_for_345 += np.eye(3) * 0.05
        geno_grm = GenotypesData(name="grm_aguilar", obs_ids=grm_obs_ids_orig, genotypes=G_original_for_345, is_grm=True)

        print("\nCalculating H-inverse for Aguilar example pedigree...")
        H_inv, H_ids = calculate_H_inverse(ped, geno_grm, weight_G=0.95)
        print(f"H_inv shape: {H_inv.shape}")
        print(f"IDs for H_inv: {H_ids}")

        np.testing.assert_allclose(H_inv.toarray(), H_inv.toarray().T, atol=1e-9, err_msg="H_inv is not symmetric.")
        print("H_inv calculation successful and symmetric.")

    except Exception as e:
        print(f"Error in H_inv calculation test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_ped_ss_path): os.remove(dummy_ped_ss_path)
```
