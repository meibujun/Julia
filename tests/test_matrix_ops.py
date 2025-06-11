import unittest
import numpy as np
from animal_breeding_genetics.matrix_ops import (
    add_matrices,
    subtract_matrices,
    multiply_matrices,
    transpose_matrix,
    invert_matrix,
    solve_linear_equations,
    eigen_decomposition,
)

class TestMatrixOps(unittest.TestCase):
    def test_add_matrices(self):
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(add_matrices(matrix_a, matrix_b), expected)

        matrix_c = np.array([[1, 2]])
        with self.assertRaises(ValueError):
            add_matrices(matrix_a, matrix_c)

    def test_subtract_matrices(self):
        matrix_a = np.array([[5, 6], [7, 8]])
        matrix_b = np.array([[1, 2], [3, 4]])
        expected = np.array([[4, 4], [4, 4]])
        np.testing.assert_array_equal(subtract_matrices(matrix_a, matrix_b), expected)

        matrix_c = np.array([[1, 2]])
        with self.assertRaises(ValueError):
            subtract_matrices(matrix_a, matrix_c)

    def test_multiply_matrices(self):
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(multiply_matrices(matrix_a, matrix_b), expected)

        matrix_c = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3
        matrix_d = np.array([[1,2],[3,4]]) # 2x2
        with self.assertRaises(ValueError):
            multiply_matrices(matrix_c, matrix_d)

    def test_transpose_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(transpose_matrix(matrix), expected)

    def test_invert_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        expected = np.array([[-2.0, 1.0], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(invert_matrix(matrix), expected)

        non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            invert_matrix(non_square_matrix)

        singular_matrix = np.array([[1, 1], [1, 1]])
        with self.assertRaises(np.linalg.LinAlgError):
            invert_matrix(singular_matrix)

    def test_solve_linear_equations(self):
        coeff_matrix = np.array([[2, 1], [1, 1]])
        constant_vector = np.array([5, 3])
        expected_solution = np.array([2, 1])
        np.testing.assert_array_almost_equal(
            solve_linear_equations(coeff_matrix, constant_vector), expected_solution
        )

        non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            solve_linear_equations(non_square_matrix, constant_vector)

        singular_matrix = np.array([[1, 1], [1, 1]])
        with self.assertRaises(np.linalg.LinAlgError):
            solve_linear_equations(singular_matrix, constant_vector)

        # Test for incompatible dimensions between coeff_matrix and constant_vector
        coeff_matrix_2 = np.array([[2, 1], [1, 1]])
        constant_vector_2 = np.array([5, 3, 4]) # Incorrect shape
        with self.assertRaises(ValueError):
            solve_linear_equations(coeff_matrix_2, constant_vector_2)


    def test_eigen_decomposition(self):
        matrix = np.array([[1, 0], [0, 2]])
        eigenvalues, eigenvectors = eigen_decomposition(matrix)
        expected_eigenvalues = np.array([1, 2])
        # Eigenvectors are not unique, so we check orthogonality and Ax = lambda*x
        np.testing.assert_array_almost_equal(eigenvalues, expected_eigenvalues)
        for i in range(len(expected_eigenvalues)):
            np.testing.assert_array_almost_equal(
                np.dot(matrix, eigenvectors[:, i]),
                expected_eigenvalues[i] * eigenvectors[:, i],
            )

        non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            eigen_decomposition(non_square_matrix)

if __name__ == "__main__":
    unittest.main()
