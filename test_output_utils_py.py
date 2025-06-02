import unittest
import pandas as pd
import numpy as np
import os
import io # For StringIO
import shutil # For robust directory removal
from output_utils_py import (
    setup_mcmc_output_files_py,
    save_mcmc_iteration_samples_py,
    close_mcmc_output_files_py,
    finalize_results_py,
    save_results_to_csv_py,
    get_ebv_py,
    save_ebv_py,
    save_summary_stats_py
)
# Use the MockMME from output_utils for testing if MME_py is not directly available/stable
from output_utils_py import MockMMEForOutput as MME_py

class MockGenotypeComponentForOutput: # Simplified for output testing
    def __init__(self, name, marker_ids, pi_value=0.5, n_markers=0):
        self.name = name
        self.marker_ids = marker_ids
        self.pi_value = pi_value # Can be float or dict for MT
        self.n_markers = n_markers if n_markers else len(marker_ids)
        # Other attributes like G, estimatePi might be needed by some output functions

class MockRandomEffectComponentForOutput:
    def __init__(self, term_array, random_type="I", variance_prior_value=1.0):
        self.term_array = term_array
        self.random_type = random_type
        # Mock variance_prior if needed by output functions (e.g. for names)
        # For now, assume output functions mainly use what's in MME_py.posterior_samples/means
        self.variance_prior = unittest.mock.Mock() # Using mock for simplicity
        self.variance_prior.value = variance_prior_value


class TestOutputUtilsPy(unittest.TestCase):

    def setUp(self):
        self.mme = MME_py() # Using the mock defined in output_utils_py or a proper one
        self.mme.lhs_vec = ["trait1"]
        self.mme.n_models = 1
        self.mme.output_id = ["id1", "id2", "id3"]

        # Mock some MCMC settings that output functions might check
        self.mme.mcmc_settings = {"outputEBV": True}
        # Mock the get_mcmc_setting method if MME_py is just a simple class here
        if not hasattr(self.mme, 'get_mcmc_setting'):
            def get_mcmc_setting_mock(key, default=None):
                return self.mme.mcmc_settings.get(key, default)
            self.mme.get_mcmc_setting = get_mcmc_setting_mock


        # Populate with some sample posterior data
        self.mme.posterior_samples = {
            'residual_variance': [1.0, 1.1, 0.9],
            'EBV_trait1': [np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25, 0.35])],
            'marker_effects_geno1_trait1': [np.array([0.01, -0.02]), np.array([0.015, -0.025])]
        }
        # Calculate means for testing finalize_results_py and get_ebv_py
        self.mme.posterior_means = {
            'residual_variance': np.mean(self.mme.posterior_samples['residual_variance']),
            'EBV_trait1': pd.DataFrame({
                "ID": self.mme.output_id, # Should match output_id
                "EBV": np.mean(np.array(self.mme.posterior_samples['EBV_trait1']), axis=0),
                "PEV_approx_SD": np.std(np.array(self.mme.posterior_samples['EBV_trait1']), axis=0)
            })
        }

        # Mock genotype components for marker effect saving
        gc1 = MockGenotypeComponentForOutput("geno1", ["m1", "m2"])
        self.mme.genotype_components.append(gc1)

        self.test_output_folder = "_test_output_py"
        if os.path.exists(self.test_output_folder):
            shutil.rmtree(self.test_output_folder)
        os.makedirs(self.test_output_folder, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_output_folder):
            shutil.rmtree(self.test_output_folder)

    def test_setup_and_save_mcmc_iteration_samples(self):
        # Test setup
        # Add a mock random effect to test its variance saving setup
        rec1 = MockRandomEffectComponentForOutput(["y1:animal"], random_type="A", variance_prior_value=np.array([[2.0]]))
        self.mme.random_effect_components.append(rec1)
        # Add polygenic variance prior to mme for setup function to find it
        self.mme.polygenic_variance_prior = rec1.variance_prior


        file_handles = setup_mcmc_output_files_py(self.mme, self.test_output_folder)
        self.assertIn("residual_variance", file_handles)
        self.assertIn("EBV_trait1", file_handles)
        self.assertIn("marker_effects_geno1_trait1", file_handles)
        self.assertIn("polygenic_effects_variance", file_handles)

        # Test saving one iteration
        current_iter_data = {
            "residual_variance": 1.05,
            "EBV_trait1": np.array([0.12, 0.22, 0.32]),
            "marker_effects_geno1_trait1": np.array([0.012, -0.022]),
            f"pi_{self.mme.genotype_components[0].name}": 0.5,
            "polygenic_effects_variance": np.array([[1.9]]) # Current sample for polygenic variance
        }
        # Need to ensure pi file and polygenic variance file are also set up if we save them
        self.assertIn(f"pi_{self.mme.genotype_components[0].name}", file_handles)


        save_mcmc_iteration_samples_py(current_iter_data, file_handles)
        close_mcmc_output_files_py(file_handles)

        # Check if files were created and have content (header + 1 data line)
        res_var_file = os.path.join(self.test_output_folder, "MCMC_samples_residual_variance.txt")
        self.assertTrue(os.path.exists(res_var_file))
        with open(res_var_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2) # Header + 1 data line
            self.assertEqual(lines[0].strip(), "trait1") # Header check
            self.assertEqual(lines[1].strip(), "1.05")

        ebv_file = os.path.join(self.test_output_folder, "MCMC_samples_EBV_trait1.txt")
        self.assertTrue(os.path.exists(ebv_file))
        with open(ebv_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0].strip(), "id1,id2,id3") # Header check
            self.assertEqual(lines[1].strip(), "0.12,0.22,0.32")

    def test_finalize_and_save_results(self):
        # finalize_results_py uses mme.posterior_samples
        results_dict = finalize_results_py(self.mme)
        self.assertIn("location_parameters", results_dict)
        self.assertIn("residual_variance", results_dict)
        self.assertTrue(isinstance(results_dict["residual_variance"], pd.DataFrame))

        ebv_key = f"EBV_{self.mme.lhs_vec[0]}"
        self.assertIn(ebv_key, results_dict)
        self.assertEqual(len(results_dict[ebv_key]), len(self.mme.output_id))


        save_results_to_csv_py(results_dict, self.test_output_folder)
        res_var_summary_file = os.path.join(self.test_output_folder, "residual_variance_summary.csv")
        self.assertTrue(os.path.exists(res_var_summary_file))
        df_check = pd.read_csv(res_var_summary_file)
        self.assertAlmostEqual(df_check["Estimate"].iloc[0], self.mme.posterior_means['residual_variance'])

    def test_get_and_save_ebv(self):
        # This test uses posterior_means which should be populated like finalize_results_py would do.
        ebv_df = get_ebv_py(self.mme)
        self.assertIsNotNone(ebv_df)
        self.assertEqual(len(ebv_df), len(self.mme.output_id) * len(self.mme.lhs_vec)) # For multi-trait if get_ebv_py concatenates
        self.assertListEqual(list(ebv_df.columns), ['ID', 'EBV', 'PEV_approx_SD', 'trait'])

        save_ebv_py(self.mme, self.test_output_folder)
        ebv_file = os.path.join(self.test_output_folder, "ebv_results.csv")
        self.assertTrue(os.path.exists(ebv_file))
        df_check = pd.read_csv(ebv_file)
        self.assertEqual(len(df_check), len(self.mme.output_id)* len(self.mme.lhs_vec))

    def test_save_summary_stats(self):
        self.mme.posterior_means["polygenic_variance"] = pd.DataFrame({
            "Covariance": ["animal_var"], "Estimate": [5.0], "SD": [0.5]
        })
        save_summary_stats_py(self.mme, self.test_output_folder)
        summary_file = os.path.join(self.test_output_folder, "mcmc_summary_statistics.txt")
        self.assertTrue(os.path.exists(summary_file))
        with open(summary_file, "r") as f:
            content = f.read()
            self.assertIn("Parameter: residual_variance", content)
            self.assertIn("Parameter: polygenic_variance", content)
            self.assertIn("animal_var", content)


if __name__ == '__main__':
    unittest.main()
