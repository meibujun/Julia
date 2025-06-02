import unittest
import pandas as pd
import numpy as np
from genostockpy.core.model_components import MME_py
from genostockpy.core.model_parser import parse_model_equation_py
from genostockpy.core.mme_builder import build_full_design_matrices_py

class TestMMEBuilder(unittest.TestCase):

    def setUp(self):
        self.mme = MME_py()
        self.pheno_data = pd.DataFrame({
            'ID': [f"id{i+1}" for i in range(5)],
            'y1': np.random.rand(5) * 10 + 20,
            'y2': np.random.rand(5) * 5 + 10,
            'age': [2.0, 3.0, 2.5, 4.0, 3.5],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'herd': ['H1', 'H2', 'H1', 'H2', 'H1']
        })
        self.mme.phenotype_dataframe = self.pheno_data
        self.mme.phenotype_info = {
            "id_column": "ID",
            "trait_columns": ["y1"], # Will be updated by parser
            "covariate_columns": ["age"]
        }
        self.mme.obs_id = self.pheno_data['ID'].tolist() # Initial obs_id

    def test_st_intercept_covariate_factor(self):
        eq = "y1 ~ intercept + age + sex"
        data_cols = self.pheno_data.columns.tolist()

        responses, terms_list, terms_map = parse_model_equation_py(
            eq, data_cols, covariate_columns_user=["age"]
        )
        self.mme.lhs_vec = responses
        self.mme.n_models = len(responses)
        self.mme.model_terms = terms_list # List of all ParsedTerm instances
        self.mme.model_term_dict = terms_map # Dict "trait:base" -> ParsedTerm

        num_total_cols = build_full_design_matrices_py(self.mme)

        self.assertIsNotNone(self.mme.X_effects_matrix)
        self.assertEqual(self.mme.X_effects_matrix.shape[0], 5 * 1) # 5 obs, 1 trait
        # Expected columns: intercept (1), age (1), sex_M (1, F is baseline if intercept present)
        # Total 3 columns if 'F' is baseline for sex.
        # pd.get_dummies for ['M','F','M','F','M'] with drop_first=True will make one col for 'M'.
        self.assertEqual(self.mme.X_effects_matrix.shape[1], 3)
        self.assertEqual(num_total_cols, 3)

        self.assertIn("y1:intercept", self.mme.effects_map)
        self.assertEqual(self.mme.effects_map["y1:intercept"]['start_col'], 0)
        self.assertEqual(self.mme.effects_map["y1:intercept"]['num_cols'], 1)

        self.assertIn("y1:age", self.mme.effects_map)
        self.assertEqual(self.mme.effects_map["y1:age"]['start_col'], 1)
        self.assertEqual(self.mme.effects_map["y1:age"]['num_cols'], 1)

        self.assertIn("y1:sex", self.mme.effects_map) # Base term for factor
        self.assertEqual(self.mme.effects_map["y1:sex"]['start_col'], 2)
        self.assertEqual(self.mme.effects_map["y1:sex"]['num_cols'], 1) # After drop_first

        # Check y_observed
        self.assertEqual(len(self.mme.y_observed), 5)
        np.testing.assert_array_almost_equal(self.mme.y_observed, self.pheno_data['y1'].values)

    def test_mt_intercept_covariate_factor(self):
        eq = "y1 ~ intercept + age; y2 ~ intercept + sex"
        data_cols = self.pheno_data.columns.tolist()

        responses, terms_list, terms_map = parse_model_equation_py(
            eq, data_cols, covariate_columns_user=["age"]
        )
        self.mme.lhs_vec = responses
        self.mme.n_models = len(responses)
        self.mme.model_terms = terms_list
        self.mme.model_term_dict = terms_map
        self.mme.phenotype_info["trait_columns"] = responses # Update for y_observed

        num_total_cols = build_full_design_matrices_py(self.mme)

        self.assertEqual(self.mme.X_effects_matrix.shape[0], 5 * 2) # 5 obs, 2 traits stacked
        # Trait1: intercept (1), age (1) = 2 cols
        # Trait2: intercept (1), sex_M (1) = 2 cols
        # Total 4 columns in X_effects_matrix due to block_diag
        self.assertEqual(self.mme.X_effects_matrix.shape[1], 4)
        self.assertEqual(num_total_cols, 4) # from fixed effects part

        # Check y_observed (stacked)
        self.assertEqual(len(self.mme.y_observed), 10)
        expected_y = np.concatenate([self.pheno_data['y1'].values, self.pheno_data['y2'].values])
        np.testing.assert_array_almost_equal(self.mme.y_observed, expected_y)

        # Check effects_map for correct start columns in the block-diagonal structure
        # y1:intercept (overall col 0), y1:age (overall col 1)
        # y2:intercept (overall col 2), y2:sex (overall col 3, assuming 1 dummy var)
        self.assertEqual(self.mme.effects_map["y1:intercept"]['start_col'], 0)
        self.assertEqual(self.mme.effects_map["y1:age"]['start_col'], 1)
        self.assertEqual(self.mme.effects_map["y2:intercept"]['start_col'], 2)
        self.assertEqual(self.mme.effects_map["y2:sex"]['start_col'], 3)


    def test_z_marker_matrix_and_map(self):
        eq = "y1 ~ intercept"
        data_cols = self.pheno_data.columns.tolist()
        responses, terms_list, terms_map = parse_model_equation_py(eq, data_cols)
        self.mme.lhs_vec = responses; self.mme.n_models = len(responses)
        self.mme.model_terms = terms_list; self.mme.model_term_dict = terms_map
        self.mme.phenotype_info["trait_columns"] = responses

        # Add a mock genotype component
        n_markers = 10
        mock_geno_matrix = np.random.randint(0,3, size=(len(self.mme.obs_id), n_markers))
        mock_gc = unittest.mock.Mock()
        mock_gc.name = "chip1"
        mock_gc.method = "BayesC0" # Indicates direct MME inclusion for Z_marker
        mock_gc.genotype_matrix = mock_geno_matrix
        mock_gc.obs_ids = self.mme.obs_id # Assume aligned
        mock_gc.marker_ids = [f"m{k}" for k in range(n_markers)]
        mock_gc.n_markers = n_markers
        self.mme.genotype_components = [mock_gc]

        num_total_cols = build_full_design_matrices_py(self.mme)

        self.assertIsNotNone(self.mme.Z_marker_matrix)
        self.assertEqual(self.mme.Z_marker_matrix.shape, (5 * 1, n_markers)) # 5 obs, 1 trait, 10 markers

        # X_effects_matrix for intercept (1 col) + Z_marker_matrix (10 cols) = 11 total effect columns
        self.assertEqual(num_total_cols, 1 + n_markers)

        marker_block_key = f"markers_{mock_gc.name}_trait_{responses[0]}" # Or similar key used by builder
        self.assertIn(marker_block_key, self.mme.effects_map)
        self.assertEqual(self.mme.effects_map[marker_block_key]['start_col'], 1) # After intercept
        self.assertEqual(self.mme.effects_map[marker_block_key]['num_cols'], n_markers)

if __name__ == '__main__':
    unittest.main()

