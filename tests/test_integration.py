# tests/test_integration.py

import unittest
import os
import pandas as pd
import numpy as np # Required for NpEncoder
import json # Import json for json.JSONEncoder
from src.agents.coordinator_agent import CoordinatorAgent
from src.core.datastructures import PhenotypeData, PedigreeData, GenotypeData # For NpEncoder and type checks

# Simplified NpEncoder, similar to the one in src/main.py
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, (PhenotypeData, PedigreeData, GenotypeData)):
             return str(obj) # Should not be directly serialized in results
        return super(NpEncoder, self).default(obj)


class TestSimpleAnalysisWorkflow(unittest.TestCase):

    def setUp(self):
        """Set up for test methods. Creates dummy CSV files."""
        self.pheno_file = 'test_dummy_phenotypes.csv'
        self.ped_file = 'test_dummy_pedigree.csv'
        self.geno_file = 'test_dummy_genotypes.csv' # For GBLUP-like test

        # Phenotype data
        pheno_df = pd.DataFrame({
            'animal_id': [1, 2, 3, 4, 1, 2, 3, 4],
            'trait_id': ['T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2'],
            'value': [10.1, 11.5, 9.8, 12.0, 101, 115, 98, 120],
            'herd': ['H1', 'H2', 'H1', 'H2', 'H1', 'H2', 'H1', 'H2'],
            'sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
        })
        pheno_df.to_csv(self.pheno_file, index=False)

        # Pedigree data
        ped_df = pd.DataFrame({
            'animal_id': [1, 2, 3, 4, 5], # Animal 5 only in pedigree
            'sire_id': [0, 0, 1, 1, 3],
            'dam_id': [0, 0, 2, 2, 4]
        })
        ped_df.to_csv(self.ped_file, index=False)

        # Genotype data (optional, for another test)
        geno_df = pd.DataFrame({
            'animal_id': [1,2,3,4], # Subset of animals
            'markerX': [0,1,2,0],
            'markerY': [1,1,0,1]
        })
        geno_df.to_csv(self.geno_file, index=False)

        self.coordinator = CoordinatorAgent()

    def tearDown(self):
        """Clean up after test methods. Removes dummy CSV files."""
        if os.path.exists(self.pheno_file):
            os.remove(self.pheno_file)
        if os.path.exists(self.ped_file):
            os.remove(self.ped_file)
        if os.path.exists(self.geno_file):
            os.remove(self.geno_file)

    def test_run_simple_animal_model_workflow(self):
        """Test the full workflow for a simple animal model."""
        model_options = {
            'target_trait': 'T1',
            'fixed_effects': ['herd', 'sex'],
            'random_effects': ['animal_additive']
        }

        results = self.coordinator.run_complete_analysis(
            phenotype_filepath=self.pheno_file,
            pedigree_filepath=self.ped_file,
            model_options=model_options
        )

        self.assertIsNotNone(results)
        self.assertIn('summary_text', results)
        self.assertIn('fixed_effects_estimates', results)
        self.assertIn('all_breeding_values', results)
        self.assertIn('top_animals_ebv', results)
        self.assertIn('variance_components', results)

        self.assertEqual(results['trait'], 'T1')

        # Check fixed effects (simulated as means)
        self.assertTrue(len(results['fixed_effects_estimates']) > 0, "Fixed effect estimates should not be empty.")
        self.assertIn('herd_H1', results['fixed_effects_estimates'])
        self.assertIn('sex_M', results['fixed_effects_estimates'])

        # Check breeding values (simulated as random)
        # Animals in pedigree: 1,2,3,4,5. Animals in T1 phenotypes: 1,2,3,4
        # ComputationalAgent creates EBVs for union of pedigree and phenotype animals.
        expected_animals_with_ebvs = {1, 2, 3, 4, 5}
        self.assertEqual(set(results['all_breeding_values'].keys()), expected_animals_with_ebvs,
                         "Mismatch in animals with EBVs.")

        # Check top_animals_ebv DataFrame
        top_ebv_df = results['top_animals_ebv']
        self.assertIsInstance(top_ebv_df, pd.DataFrame)
        if not top_ebv_df.empty:
            self.assertIn('animal_id', top_ebv_df.columns)
            self.assertIn('ebv', top_ebv_df.columns)
            self.assertTrue(len(top_ebv_df) <= 5) # Default top_n_ebv is 5 in ResultsAnalysisAgent

        # Check variance components (simulated)
        self.assertIn('animal_additive', results['variance_components'])
        self.assertIn('residual', results['variance_components'])

        print("\nIntegration Test (Animal Model) Summary Text:")
        print(results['summary_text'])

    def test_run_gblup_like_model_workflow(self):
        """Test the full workflow for a GBLUP-like model (with genotype data)."""
        model_options = {
            'target_trait': 'T2', # Using T2 for this test
            'fixed_effects': ['herd'],
            'random_effects': ['animal_additive', 'genomic'] # Indicate genomic component
        }

        results = self.coordinator.run_complete_analysis(
            phenotype_filepath=self.pheno_file,
            pedigree_filepath=self.ped_file,
            model_options=model_options,
            genotype_filepath_data=self.geno_file
            # No marker info file in this simple test
        )

        self.assertIsNotNone(results)
        self.assertEqual(results['trait'], 'T2')
        self.assertIn('summary_text', results)
        self.assertIn('variance_components', results)
        # Check if genomic variance component was added (as per ComputationalAgent simulation)
        self.assertIn('genomic_animal_effect', results['variance_components'],
                      "Genomic variance component missing for GBLUP-like model.")

        # Animals in T2 phenotypes: 1,2,3,4. Animals in pedigree: 1,2,3,4,5.
        # Animals in genotype data: 1,2,3,4
        expected_animals_with_ebvs = {1, 2, 3, 4, 5}
        self.assertEqual(set(results['all_breeding_values'].keys()), expected_animals_with_ebvs,
                         "Mismatch in animals with EBVs for GBLUP-like model.")

        print("\nIntegration Test (GBLUP-like Model) Summary Text:")
        print(results['summary_text'])


if __name__ == '__main__':
    unittest.main()
