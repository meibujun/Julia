import pytest
import numpy as np
import pandas as pd
import os

# Assuming pyjwas is installed or PYTHONPATH is set up correctly
from pyjwas.data import PhenotypeData, GenotypeData

# Determine the directory of this test file to build paths to data files
current_dir = os.path.dirname(os.path.abspath(__file__))
TEST_PHENOTYPES_CSV = os.path.join(current_dir, "data", "test_phenotypes.csv")
TEST_GENOTYPES_CSV = os.path.join(current_dir, "data", "test_genotypes.csv")

def test_phenotype_data_loading():
    """Tests loading phenotype data, accessing attributes, and y-vector."""
    assert os.path.exists(TEST_PHENOTYPES_CSV), f"Test phenotype file not found at {TEST_PHENOTYPES_CSV}"
    
    pheno_data = PhenotypeData(file_path=TEST_PHENOTYPES_CSV)
    
    assert pheno_data.data is not None, "PhenotypeData.data should not be None after loading."
    assert isinstance(pheno_data.data, pd.DataFrame), "PhenotypeData.data should be a pandas DataFrame."
    assert len(pheno_data.data) == 4, "Should have 4 phenotype records."
    
    expected_ids = ["p1", "p2", "p3", "p4"]
    assert pheno_data.individual_ids is not None, "PhenotypeData.individual_ids should not be None."
    assert pheno_data.individual_ids == expected_ids, f"Phenotype IDs do not match. Expected {expected_ids}, got {pheno_data.individual_ids}"
    
    y_vector = pheno_data.get_y_vector('y1')
    assert y_vector is not None, "get_y_vector('y1') should return a vector."
    assert isinstance(y_vector, np.ndarray), "y_vector should be a NumPy array."
    assert len(y_vector) == 4, "y_vector should have 4 elements."
    expected_y = np.array([10.5, 12.0, 9.8, 11.5])
    assert np.allclose(y_vector, expected_y), f"y_vector values are incorrect. Expected {expected_y}, got {y_vector}"

    with pytest.raises(ValueError, match="Column 'non_existent_col' not found"):
        pheno_data.get_y_vector('non_existent_col')

def test_genotype_data_loading():
    """Tests loading genotype data and accessing attributes."""
    assert os.path.exists(TEST_GENOTYPES_CSV), f"Test genotype file not found at {TEST_GENOTYPES_CSV}"
    
    geno_data = GenotypeData(file_path=TEST_GENOTYPES_CSV)
    
    assert geno_data.genotype_matrix is not None, "GenotypeData.genotype_matrix should not be None."
    assert isinstance(geno_data.genotype_matrix, np.ndarray), "GenotypeData.genotype_matrix should be a NumPy array."
    # 5 individuals (p1, p2, p3, p4, g5) and 5 markers (m1 to m5)
    assert geno_data.genotype_matrix.shape == (5, 5), f"Genotype matrix shape is incorrect. Expected (5, 5), got {geno_data.genotype_matrix.shape}"
    
    expected_ids = ["p1", "p2", "p3", "p4", "g5"]
    assert geno_data.individual_ids is not None, "GenotypeData.individual_ids should not be None."
    assert geno_data.individual_ids == expected_ids, f"Genotype IDs do not match. Expected {expected_ids}, got {geno_data.individual_ids}"
    
    # Check if data is numeric (it should be after loading)
    assert np.issubdtype(geno_data.genotype_matrix.dtype, np.number), "Genotype matrix data type should be numeric."

def test_grm_calculation():
    """Tests GRM calculation logic."""
    assert os.path.exists(TEST_GENOTYPES_CSV), f"Test genotype file not found at {TEST_GENOTYPES_CSV}"
    
    geno_data = GenotypeData(file_path=TEST_GENOTYPES_CSV)
    grm = geno_data.calculate_grm()
    
    assert grm is not None, "GRM should not be None after calculation."
    assert isinstance(grm, np.ndarray), "GRM should be a NumPy array."
    
    num_individuals = len(geno_data.individual_ids) # Should be 5
    assert grm.shape == (num_individuals, num_individuals), f"GRM shape is incorrect. Expected ({num_individuals}, {num_individuals}), got {grm.shape}"
    
    assert np.allclose(grm, grm.T, atol=1e-7), "GRM should be symmetric."
    
    # Optional: Check diagonal elements (usually around 1, but can vary based on coding and allele freqs)
    # For VanRaden, diagonal elements are related to individual homozygosity relative to average.
    # This is a basic check, not a strict validation of values without knowing the exact expected GRM.
    assert np.all(np.diag(grm) > 0), "Diagonal elements of GRM should generally be positive."

    # Test with a very simple case (manual calculation possible)
    # Create a temporary CSV for this specific test
    simple_geno_content = "ID,m1,m2\nind1,0,2\nind2,2,0"
    simple_geno_file = os.path.join(current_dir, "data", "simple_genotypes.csv")
    with open(simple_geno_file, "w") as f:
        f.write(simple_geno_content)
    
    simple_geno_data = GenotypeData(simple_geno_file)
    simple_grm = simple_geno_data.calculate_grm()

    # M = [[0, 2], [2, 0]]
    # p_m1 = (0+2)/(2*2) = 0.5; p_m2 = (2+0)/(2*2) = 0.5
    # P_m1 = 2*0.5 = 1; P_m2 = 2*0.5 = 1
    # P = [[1, 1], [1, 1]]
    # Z = M - P = [[-1, 1], [1, -1]]
    # Sum(2*p_j*(1-p_j)) = 2*0.5*(1-0.5) + 2*0.5*(1-0.5) = 0.5 + 0.5 = 1.0
    # Denom = 1.0
    # ZZ_T = [[-1, 1], [1, -1]] @ [[-1, 1], [1, -1]].T = [[-1, 1], [1, -1]] @ [[-1, 1], [1, -1]]
    #      = [[2, -2], [-2, 2]]
    # G = ZZ_T / Denom = [[2, -2], [-2, 2]]
    expected_simple_grm = np.array([[2.0, -2.0], [-2.0, 2.0]])
    assert np.allclose(simple_grm, expected_simple_grm), f"Simple GRM calculation is incorrect. Expected {expected_simple_grm}, got {simple_grm}"
    
    os.remove(simple_geno_file) # Clean up

# To run these tests, navigate to the directory containing `pyjwas` 
# and run `pytest pyjwas/tests/test_data.py`
# Ensure pyjwas is in PYTHONPATH or installed. Example: export PYTHONPATH=$PYTHONPATH:$(pwd)
# (from the directory containing the pyjwas folder)
