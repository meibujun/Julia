import numpy as np

def add_matrices(matrix_a, matrix_b):
    """
    Adds two matrices.

    Args:
        matrix_a (np.ndarray): The first matrix.
        matrix_b (np.ndarray): The second matrix.

    Returns:
        np.ndarray: The sum of the two matrices.

    Raises:
        ValueError: If the matrices have incompatible shapes.
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Matrices must have the same shape for addition.")
    return np.add(matrix_a, matrix_b)

def subtract_matrices(matrix_a, matrix_b):
    """
    Subtracts matrix_b from matrix_a.

    Args:
        matrix_a (np.ndarray): The matrix from which to subtract.
        matrix_b (np.ndarray): The matrix to subtract.

    Returns:
        np.ndarray: The result of the subtraction.

    Raises:
        ValueError: If the matrices have incompatible shapes.
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Matrices must have the same shape for subtraction.")
    return np.subtract(matrix_a, matrix_b)

def multiply_matrices(matrix_a, matrix_b):
    """
    Multiplies two matrices (dot product).

    Args:
        matrix_a (np.ndarray): The first matrix.
        matrix_b (np.ndarray): The second matrix.

    Returns:
        np.ndarray: The product of the two matrices.

    Raises:
        ValueError: If the matrices have incompatible shapes for multiplication.
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            "Number of columns in matrix_a must equal number of rows in matrix_b."
        )
    return np.dot(matrix_a, matrix_b)

def transpose_matrix(matrix):
    """
    Transposes a matrix.

    Args:
        matrix (np.ndarray): The matrix to transpose.

    Returns:
        np.ndarray: The transposed matrix.
    """
    return np.transpose(matrix)

def invert_matrix(matrix):
    """
    Inverts a square matrix.

    Args:
        matrix (np.ndarray): The matrix to invert.

    Returns:
        np.ndarray: The inverted matrix.

    Raises:
        ValueError: If the matrix is not square.
        np.linalg.LinAlgError: If the matrix is singular and not invertible.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for inversion.")
    return np.linalg.inv(matrix)

def solve_linear_equations(coeff_matrix, constant_vector):
    """
    Solves a system of linear equations Ax = b.

    Args:
        coeff_matrix (np.ndarray): The coefficient matrix (A).
        constant_vector (np.ndarray): The constant vector (b).

    Returns:
        np.ndarray: The solution vector (x).

    Raises:
        ValueError: If the coefficient matrix is not square or if the
                    dimensions of the matrix and vector are incompatible.
        np.linalg.LinAlgError: If the coefficient matrix is singular.
    """
    if coeff_matrix.shape[0] != coeff_matrix.shape[1]:
        raise ValueError("Coefficient matrix must be square.")
    if coeff_matrix.shape[0] != constant_vector.shape[0]:
        raise ValueError(
            "Number of rows in coefficient matrix must equal number of elements in constant vector."
        )
    return np.linalg.solve(coeff_matrix, constant_vector)

def eigen_decomposition(matrix):
    """
    Computes eigenvalues and eigenvectors of a matrix.

    Args:
        matrix (np.ndarray): The matrix for which to compute eigenvalues/eigenvectors.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Eigenvalues.
            - np.ndarray: Eigenvectors.

    Raises:
        ValueError: If the matrix is not square.
        np.linalg.LinAlgError: If the eigenvalue computation does not converge.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for eigen decomposition.")
    return np.linalg.eig(matrix)
