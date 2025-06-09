# tests/test_data_management/test_io_handlers.py

import unittest
import pandas as pd
import numpy as np
import tempfile
import os

# Adjust path to import from the root of the project
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.data_management.io_handlers import (
    read_phenotypic_data,
    read_pedigree_data,
    save_data_to_csv,
    read_genomic_data,
    save_genomic_data
)
from sheep_breeding_genomics.data_management.data_structures import GenomicData


class TestIOHandlers(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to store test CSV files
        self.test_dir = tempfile.TemporaryDirectory()
        self.pheno_file_path = os.path.join(self.test_dir.name, 'test_pheno.csv')
        self.ped_file_path = os.path.join(self.test_dir.name, 'test_ped.csv')
        self.geno_file_path = os.path.join(self.test_dir.name, 'test_geno.csv')

        # Sample data
        self.pheno_df = pd.DataFrame({'AnimalID': [1, 2, 3], 'TraitA': [10.1, 10.2, 10.3]})
        self.ped_df = pd.DataFrame({'AnimalID': [1, 2, 3], 'SireID': [0, 1, 1], 'DamID': [0, 0, 2]})
        self.geno_raw_df = pd.DataFrame({
            'ChipID': ['id1', 'id2', 'id3'],
            'SNP1': [0, 1, 2],
            'SNP2': [1, 1, np.nan], # Add a NaN for realistic scenario
            'SNP3': [2, 0, 1]
        })


    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    # --- Tests for Phenotypic Data ---
    def test_save_and_read_phenotypic_data(self):
        # Test saving
        success = save_data_to_csv(self.pheno_df, self.pheno_file_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.pheno_file_path))

        # Test reading
        read_df = read_phenotypic_data(self.pheno_file_path)
        pd.testing.assert_frame_equal(self.pheno_df, read_df)

    def test_read_phenotypic_data_non_existent_file(self):
        read_df = read_phenotypic_data("non_existent_file.csv")
        self.assertTrue(read_df.empty) # Expect empty DataFrame on error

    def test_save_empty_phenotypic_data(self):
        empty_df = pd.DataFrame()
        success = save_data_to_csv(empty_df, self.pheno_file_path) # This should print warning and return False
        self.assertFalse(success)
        self.assertFalse(os.path.exists(self.pheno_file_path)) # File should not be created


    # --- Tests for Pedigree Data ---
    def test_save_and_read_pedigree_data(self):
        success = save_data_to_csv(self.ped_df, self.ped_file_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.ped_file_path))

        read_df = read_pedigree_data(self.ped_file_path)
        pd.testing.assert_frame_equal(self.ped_df, read_df)

    def test_read_pedigree_data_non_existent_file(self):
        read_df = read_pedigree_data("non_existent_file.csv")
        self.assertTrue(read_df.empty)


    # --- Tests for Genomic Data ---
    def test_save_and_read_genomic_data(self):
        animal_id_col = 'ChipID'
        # Test saving (using the generic save_data_to_csv for now, or specialized if exists)
        # Assuming save_genomic_data is a wrapper or similar to save_data_to_csv for DataFrames
        success_save = save_genomic_data(self.geno_raw_df, self.geno_file_path, animal_id_col=animal_id_col)
        self.assertTrue(success_save)
        self.assertTrue(os.path.exists(self.geno_file_path))

        # Test reading
        loaded_geno_data_obj = read_genomic_data(self.geno_file_path, animal_id_col=animal_id_col)
        self.assertIsNotNone(loaded_geno_data_obj)
        self.assertIsInstance(loaded_geno_data_obj, GenomicData)
        # Pandas by default reads NaN as float, ensure original int columns with NaN are handled.
        # Our GenomicData structure just wraps the DataFrame, so comparison should be fine.
        # For read_csv, integers with NaNs become float. This is expected pandas behavior.
        expected_df = self.geno_raw_df.copy() # Make a copy to modify for expected type changes
        expected_df['SNP2'] = expected_df['SNP2'].astype(float) # SNP2 has NaN, so it will be float

        pd.testing.assert_frame_equal(loaded_geno_data_obj.data, expected_df)
        self.assertEqual(loaded_geno_data_obj.animal_id_col, animal_id_col)

    def test_read_genomic_data_non_existent_file(self):
        geno_data_obj = read_genomic_data("non_existent_geno.csv", animal_id_col='AnimalID')
        self.assertIsNone(geno_data_obj) # Expect None on error

    def test_read_genomic_data_missing_animal_id_col_in_file(self):
        # Save a file without the specified animal_id_col
        temp_df = pd.DataFrame({'SNP_X': [1,2,3]})
        temp_df.to_csv(self.geno_file_path, index=False)

        geno_data_obj = read_genomic_data(self.geno_file_path, animal_id_col='AnimalID') # AnimalID is not in file
        self.assertIsNone(geno_data_obj)

    def test_save_empty_genomic_data_not_allowed(self):
        empty_df = pd.DataFrame(columns=['ChipID', 'SNP1']) # Has AnimalID col but no data
        success = save_genomic_data(empty_df, self.geno_file_path, animal_id_col='ChipID', allow_empty=False)
        self.assertFalse(success) # Should not save if no actual animal records and allow_empty=False
        self.assertFalse(os.path.exists(self.geno_file_path))

    def test_save_empty_genomic_data_allowed(self):
        empty_df = pd.DataFrame(columns=['ChipID', 'SNP1'])
        success = save_genomic_data(empty_df, self.geno_file_path, animal_id_col='ChipID', allow_empty=True)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.geno_file_path))

        # Check if it reads back as an empty GenomicData object's DataFrame
        loaded_empty_geno = read_genomic_data(self.geno_file_path, animal_id_col='ChipID')
        self.assertIsNotNone(loaded_empty_geno)
        self.assertTrue(loaded_empty_geno.data.empty) # pd.read_csv on empty file makes empty df

    def test_save_genomic_data_no_snp_cols_not_allowed(self):
        id_only_df = pd.DataFrame({'ChipID': ['id1', 'id2']})
        success = save_genomic_data(id_only_df, self.geno_file_path, animal_id_col='ChipID', allow_empty=False)
        self.assertFalse(success)
        self.assertFalse(os.path.exists(self.geno_file_path))

    def test_save_genomic_data_no_snp_cols_allowed(self):
        id_only_df = pd.DataFrame({'ChipID': ['id1', 'id2']})
        success = save_genomic_data(id_only_df, self.geno_file_path, animal_id_col='ChipID', allow_empty=True)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.geno_file_path))
        loaded_id_only_geno = read_genomic_data(self.geno_file_path, animal_id_col='ChipID')
        self.assertIsNotNone(loaded_id_only_geno)
        pd.testing.assert_frame_equal(loaded_id_only_geno.data, id_only_df)


if __name__ == '__main__':
    unittest.main()
