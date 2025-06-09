# tests/test_data_management/test_validators.py

import unittest
import pandas as pd
import numpy as np

# Adjust path to import from the root of the project
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.data_management.data_structures import PhenotypicData, PedigreeData, GenomicData
from sheep_breeding_genomics.data_management.validators import (
    validate_phenotypic_data,
    validate_pedigree_data,
    validate_genomic_data,
    filter_by_call_rate_snps,
    filter_by_call_rate_animals,
    filter_by_maf
)

class TestPhenotypicDataValidator(unittest.TestCase):

    def test_validate_phenotypic_data_valid(self):
        df = pd.DataFrame({'AnimalID': [1, 2, 3], 'Age': [24, 30, 22], 'Weight': [50.5, 60.2, 55.0]})
        phen_data = PhenotypicData(df)
        is_valid = validate_phenotypic_data(phen_data, required_columns=['Age', 'Weight'], expected_dtypes={'Age': int, 'Weight': float})
        self.assertTrue(is_valid)

    def test_validate_phenotypic_data_missing_animalid(self):
        df = pd.DataFrame({'Age': [24, 30, 22], 'Weight': [50.5, 60.2, 55.0]}) # Missing AnimalID
        phen_data = PhenotypicData(df) # This itself won't fail, but validator should
        is_valid = validate_phenotypic_data(phen_data)
        self.assertFalse(is_valid) # Expect AnimalID missing error

    def test_validate_phenotypic_data_duplicate_animalid(self):
        df = pd.DataFrame({'AnimalID': [1, 1, 2], 'Weight': [50, 51, 52]})
        phen_data = PhenotypicData(df)
        is_valid = validate_phenotypic_data(phen_data) # Should print warning, but return True
        self.assertTrue(is_valid)

    def test_validate_phenotypic_data_missing_required_column(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'Weight': [50.5, 60.2]})
        phen_data = PhenotypicData(df)
        is_valid = validate_phenotypic_data(phen_data, required_columns=['Age'])
        self.assertFalse(is_valid)

    def test_validate_phenotypic_data_wrong_dtype_no_cast(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'Age': ["24", "30"]}) # Age as string
        phen_data = PhenotypicData(df)
        # If we expect int, and it's object due to strings, it should try to cast or fail
        # The current validator attempts astype(dtype).
        is_valid = validate_phenotypic_data(phen_data, expected_dtypes={'Age': int})
        self.assertTrue(is_valid) # Cast should succeed
        self.assertTrue(pd.api.types.is_integer_dtype(phen_data.data['Age']))

    def test_validate_phenotypic_data_wrong_dtype_cast_fail(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'Age': ["24a", "30b"]}) # Age as string, not castable to int
        phen_data = PhenotypicData(df)
        is_valid = validate_phenotypic_data(phen_data, expected_dtypes={'Age': int})
        self.assertFalse(is_valid)


class TestPedigreeDataValidator(unittest.TestCase):

    def test_validate_pedigree_data_valid(self):
        df = pd.DataFrame({'AnimalID': [1,2,3,4], 'SireID': [0,1,1,0], 'DamID': [0,0,2,2]})
        ped_data = PedigreeData(df)
        is_valid = validate_pedigree_data(ped_data, check_loops=True)
        self.assertTrue(is_valid)

    def test_validate_pedigree_data_missing_animalid_value(self):
        df = pd.DataFrame({'AnimalID': [1, np.nan, 3], 'SireID': [0,1,0], 'DamID': [0,0,2]})
        ped_data = PedigreeData(df) # This should be fine for PedigreeData init if column exists
        is_valid = validate_pedigree_data(ped_data)
        self.assertFalse(is_valid) # Validator should catch NaN in AnimalID

    def test_validate_pedigree_data_duplicate_animalid(self):
        df = pd.DataFrame({'AnimalID': [1,1,2], 'SireID': [0,0,1], 'DamID': [0,0,0]})
        ped_data = PedigreeData(df)
        is_valid = validate_pedigree_data(ped_data)
        self.assertFalse(is_valid) # Validator should catch duplicate AnimalIDs

    def test_validate_pedigree_data_parent_not_in_pedigree(self):
        df = pd.DataFrame({'AnimalID': [1,2], 'SireID': [0,3], 'DamID': [0,0]}) # Sire 3 does not exist
        ped_data = PedigreeData(df)
        is_valid = validate_pedigree_data(ped_data)
        self.assertFalse(is_valid)

    def test_validate_pedigree_data_loop(self):
        df = pd.DataFrame({'AnimalID': [1,2], 'SireID': [0,1], 'DamID': [0,1]}) # Animal 1 is sire and dam of 2
        ped_data = PedigreeData(df)
        is_valid_no_loop_check = validate_pedigree_data(ped_data, check_loops=False)
        self.assertTrue(is_valid_no_loop_check) # Valid if not checking loops specifically for sire==dam

        # The current loop check is for animal being its own sire/dam, or sire==dam
        # Animal 1 is sire AND dam of 2. This is a specific type of "loop" or issue.
        # The warning "AnimalID '2' has identical SireID and DamID: '1'." should appear.
        # The function should still return True as it's a warning.
        is_valid_with_loop_check = validate_pedigree_data(ped_data, check_loops=True)
        self.assertTrue(is_valid_with_loop_check) # This specific case is a warning, not error

        df_self_sire = pd.DataFrame({'AnimalID': [1], 'SireID': [1], 'DamID': [0]})
        ped_data_self_sire = PedigreeData(df_self_sire)
        is_valid_self_sire = validate_pedigree_data(ped_data_self_sire, check_loops=True)
        self.assertFalse(is_valid_self_sire) # Animal cannot be its own sire


class TestGenomicDataValidator(unittest.TestCase):
    def setUp(self):
        self.sample_geno_df = pd.DataFrame({
            'AnimalID': ['A1', 'A2', 'A3', 'A4'],
            'SNP1': [0, 1, 2, 0],
            'SNP2': [1, np.nan, 0, 2], # Has NaN
            'SNP3': [0, 0, 0, 0]      # Monomorphic
        })
        self.gen_data = GenomicData(self.sample_geno_df)

    def test_validate_genomic_data_valid(self):
        is_valid, snp_stats_df = validate_genomic_data(self.gen_data)
        self.assertTrue(is_valid)
        self.assertFalse(snp_stats_df.empty)
        self.assertEqual(len(snp_stats_df), 3) # 3 SNPs
        # Check SNP2 call rate: 3 animals called / 4 total animals = 0.75
        self.assertAlmostEqual(snp_stats_df[snp_stats_df['snp'] == 'SNP2']['call_rate'].iloc[0], 0.75)
        # Check SNP3 MAF (monomorphic)
        self.assertAlmostEqual(snp_stats_df[snp_stats_df['snp'] == 'SNP3']['maf'].iloc[0], 0.0)


    def test_validate_genomic_data_invalid_genotypes(self):
        df_invalid = self.sample_geno_df.copy()
        df_invalid.loc[0, 'SNP1'] = 3 # Invalid genotype
        gen_data_invalid = GenomicData(df_invalid)
        is_valid, _ = validate_genomic_data(gen_data_invalid)
        self.assertFalse(is_valid)

    def test_filter_by_call_rate_snps(self):
        # SNP2 call rate = 0.75. SNP1, SNP3 call rate = 1.0
        filtered_geno_data = filter_by_call_rate_snps(self.gen_data, min_call_rate=0.8)
        self.assertEqual(filtered_geno_data.num_snps, 2) # SNP1, SNP3 should remain
        self.assertNotIn('SNP2', filtered_geno_data.snp_names)

    def test_filter_by_call_rate_animals(self):
        # Animal A2 has 1 NaN out of 3 SNPs -> call rate = 2/3 = 0.667
        # Others have 1.0 call rate.
        filtered_geno_data = filter_by_call_rate_animals(self.gen_data, min_call_rate=0.9)
        self.assertEqual(filtered_geno_data.num_animals, 3) # A1, A3, A4 should remain
        self.assertNotIn('A2', filtered_geno_data.animal_ids)

    def test_filter_by_maf(self):
        # SNP1: p = (1+2)/(2*4)=3/8=0.375. MAF = 0.375
        # SNP2: (1+0+2)/(2*3)=3/6=0.5 (among called). MAF = 0.5
        # SNP3: MAF = 0
        filtered_geno_data = filter_by_maf(self.gen_data, min_maf=0.1)
        self.assertEqual(filtered_geno_data.num_snps, 2) # SNP1, SNP2 should remain
        self.assertNotIn('SNP3', filtered_geno_data.snp_names)

    def test_filter_by_maf_no_called_genotypes(self):
        df_all_nan = pd.DataFrame({
            'AnimalID': ['A1', 'A2'],
            'SNP_ALL_NAN': [np.nan, np.nan]
        })
        gen_data_all_nan = GenomicData(df_all_nan)
        # MAF will be NaN. Should be kept by default by filter_by_maf.
        filtered_data = filter_by_maf(gen_data_all_nan, min_maf=0.01)
        self.assertEqual(filtered_data.num_snps, 1)
        self.assertIn('SNP_ALL_NAN', filtered_data.snp_names)


if __name__ == '__main__':
    unittest.main()
