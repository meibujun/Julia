# tests/test_genetic_evaluation/test_blup_models.py

import unittest
import pandas as pd
import numpy as np

# Adjust path to import from the root of the project
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.data_management.data_structures import PhenotypicData, GenomicData, PedigreeData
from sheep_breeding_genomics.genetic_evaluation.relationship_matrix import calculate_nrm, calculate_grm, calculate_h_inverse_matrix
from sheep_breeding_genomics.genetic_evaluation.blup_models import solve_animal_model_mme, solve_ssgblup_model_mme

class TestBlupModels(unittest.TestCase):

    def setUp(self):
        # Common data for tests
        self.animal_id_col = 'AnimalID'
        self.trait_col = 'TraitValue'

        # Phenotypic Data
        self.pheno_data_dict = {
            self.animal_id_col: ['A1', 'A2', 'A3', 'A4', 'N1'],
            self.trait_col: [10.5, 12.0, 9.8, 11.5, 10.0],
            'FixedEffect1': ['G1', 'G2', 'G1', 'G2', 'G1']
        }
        self.phenotypic_df = pd.DataFrame(self.pheno_data_dict)

        # Pedigree Data for NRM
        self.ped_data_dict = {
            'AnimalID': ['A1', 'A2', 'A3', 'A4', 'N1', 'P1', 'P2'], # P1, P2 are parents
            'SireID':   ['P1', 'P1', 'A1', 'A1', '0', '0', '0'],
            'DamID':    ['P2', 'P2', 'A2', 'A2', '0', '0', '0']
        }
        self.pedigree_df = pd.DataFrame(self.ped_data_dict)
        self.nrm_df = calculate_nrm(self.pedigree_df, founder_val='0')
        # Ensure NRM is invertible for tests by adding small value to diagonal
        if not self.nrm_df.empty:
            np.fill_diagonal(self.nrm_df.values, np.diag(self.nrm_df.values) + 1e-6)


        # Genomic Data for GRM (animals A1, A2, A3, A4)
        self.geno_data_dict = {
            'AnimalID': ['A1', 'A2', 'A3', 'A4'],
            'SNP1': [0,1,2,0], 'SNP2': [1,1,0,2], 'SNP3': [2,0,1,1]
        }
        self.genomic_data_obj = GenomicData(pd.DataFrame(self.geno_data_dict))
        self.grm_df = calculate_grm(self.genomic_data_obj)
        # Ensure GRM is invertible
        if not self.grm_df.empty:
             np.fill_diagonal(self.grm_df.values, np.diag(self.grm_df.values) + 1e-6)


        # H_inv for ssGBLUP (using all animals from NRM, genotyped A1-A4)
        if not self.nrm_df.empty and not self.grm_df.empty:
            self.h_inv_df = calculate_h_inverse_matrix(self.nrm_df, self.grm_df)
        else:
            self.h_inv_df = pd.DataFrame() # Placeholder if setup fails

        # Variance components
        self.var_animal = 4.0
        self.var_residual = 12.0 # h2 = 4 / (4+12) = 0.25

    def test_solve_animal_model_mme_pblup(self):
        if self.nrm_df.empty:
            self.skipTest("NRM calculation failed in setUp, skipping PBLUP test.")

        ebv_df = solve_animal_model_mme(
            phenotypic_df=self.phenotypic_df,
            relationship_matrix_df=self.nrm_df,
            trait_col=self.trait_col,
            animal_id_col=self.animal_id_col,
            var_animal=self.var_animal,
            var_residual=self.var_residual,
            fixed_effects_cols=[] # Intercept only
        )
        self.assertFalse(ebv_df.empty)
        self.assertEqual(len(ebv_df), len(self.nrm_df))
        self.assertIn('EBV', ebv_df.columns)
        self.assertIn(self.animal_id_col, ebv_df.columns)

    def test_solve_animal_model_mme_gblup(self):
        if self.grm_df.empty:
            self.skipTest("GRM calculation failed in setUp, skipping GBLUP test.")

        # Phenotypes only for genotyped animals for a clean GBLUP
        pheno_gblup_df = self.phenotypic_df[self.phenotypic_df[self.animal_id_col].isin(self.grm_df.index)]

        gebv_df = solve_animal_model_mme(
            phenotypic_df=pheno_gblup_df,
            relationship_matrix_df=self.grm_df, # Use GRM
            trait_col=self.trait_col,
            animal_id_col=self.animal_id_col,
            var_animal=self.var_animal, # Interpreted as var_genomic
            var_residual=self.var_residual,
            fixed_effects_cols=[]
        )
        self.assertFalse(gebv_df.empty)
        self.assertEqual(len(gebv_df), len(self.grm_df))
        self.assertIn('EBV', gebv_df.columns) # Column is named 'EBV'

    def test_solve_animal_model_mme_with_fixed_effect(self):
        if self.nrm_df.empty:
            self.skipTest("NRM calculation failed, skipping fixed effect test.")

        ebv_df_fe = solve_animal_model_mme(
            phenotypic_df=self.phenotypic_df,
            relationship_matrix_df=self.nrm_df,
            trait_col=self.trait_col,
            animal_id_col=self.animal_id_col,
            var_animal=self.var_animal,
            var_residual=self.var_residual,
            fixed_effects_cols=['FixedEffect1'] # Add a fixed effect
        )
        self.assertFalse(ebv_df_fe.empty)
        # Fixed effects are printed, not returned with EBVs. Check number of EBVs.
        self.assertEqual(len(ebv_df_fe), len(self.nrm_df))


    def test_solve_ssgblup_model_mme(self):
        if self.h_inv_df.empty:
            self.skipTest("H_inv calculation failed in setUp, skipping ssGBLUP test.")

        ssgebv_df = solve_ssgblup_model_mme(
            phenotypic_df=self.phenotypic_df, # Use phenotypes for all animals (N1 has pheno)
            h_inv_df=self.h_inv_df,
            trait_col=self.trait_col,
            animal_id_col=self.animal_id_col,
            var_genetic=self.var_animal, # var_genetic for ssGBLUP
            var_residual=self.var_residual,
            fixed_effects_cols=[]
        )
        self.assertFalse(ssgebv_df.empty)
        self.assertEqual(len(ssgebv_df), len(self.h_inv_df)) # EBVs for all animals in H_inv
        self.assertIn('ssGEBV', ssgebv_df.columns)

    def test_mme_solvers_missing_variances(self):
        # Test PBLUP with missing variances (should use defaults)
        if self.nrm_df.empty:
            self.skipTest("NRM calculation failed, skipping missing var test for PBLUP.")
        ebv_df_pblup_default_var = solve_animal_model_mme(
            phenotypic_df=self.phenotypic_df,
            relationship_matrix_df=self.nrm_df,
            trait_col=self.trait_col,
            animal_id_col=self.animal_id_col
        )
        self.assertFalse(ebv_df_pblup_default_var.empty)

        # Test ssGBLUP with missing variances
        if self.h_inv_df.empty:
            self.skipTest("H_inv calculation failed, skipping missing var test for ssGBLUP.")
        ssgebv_df_default_var = solve_ssgblup_model_mme(
            phenotypic_df=self.phenotypic_df,
            h_inv_df=self.h_inv_df,
            trait_col=self.trait_col,
            animal_id_col=self.animal_id_col
        )
        self.assertFalse(ssgebv_df_default_var.empty)

if __name__ == '__main__':
    unittest.main()
