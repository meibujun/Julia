import unittest
import pandas as pd
import numpy as np
import os
import shutil
import pathlib

from genostockpy.api import GenoStockModel

class TestIntegrationBayesC0(unittest.TestCase):

    def setUp(self):
        self.base_test_dir = pathlib.Path("_test_integration_data_bayesc0")
        self.data_files_dir = self.base_test_dir / "data" / "bayesc0_example"
        os.makedirs(self.data_files_dir, exist_ok=True)

        # Results are written to /tmp by the test method now
        self.results_capture_file = pathlib.Path("/tmp/bayes_results.txt")
        if self.results_capture_file.exists():
            os.remove(self.results_capture_file)

        self.n_ind = 20
        self.pheno_data = pd.DataFrame({
            'ID': [f"id{i+1}" for i in range(self.n_ind)],
            'yield': np.random.normal(loc=30, scale=5, size=self.n_ind),
            'sex': np.random.choice(['M', 'F'], size=self.n_ind)
        })
        self.pheno_file = self.data_files_dir / "phenotypes.csv"
        self.pheno_data.to_csv(self.pheno_file, index=False)

        self.n_markers = 50
        geno_matrix = np.random.randint(0, 3, size=(self.n_ind, self.n_markers))
        marker_ids = [f"m{j+1}" for j in range(self.n_markers)]
        geno_data_df = pd.DataFrame(geno_matrix, columns=marker_ids)
        geno_data_df.insert(0, 'ID', [f"id{i+1}" for i in range(self.n_ind)])
        self.geno_file = self.data_files_dir / "genotypes.csv"
        geno_data_df.to_csv(self.geno_file, index=False)

    def tearDown(self):
        if self.base_test_dir.exists():
            shutil.rmtree(self.base_test_dir)
        # if self.results_capture_file.exists(): # Keep the capture file for inspection by `read_files`
        #     os.remove(self.results_capture_file)


    def test_bayesc0_run_simple_model(self):
        model = GenoStockModel(model_name="BayesC0_IntegrationTest")

        model.set_model_equation(
            equation="yield = intercept + sex",
            trait_types={"yield": "continuous"}
        )

        model.load_phenotypes(
            data=str(self.pheno_file),
            id_column="ID"
        )

        model.add_genotypes(
            name="chip_data",
            data_source=str(self.geno_file),
            method="BayesC0",
            genetic_variance_value=1.0,
            df_prior=5.0,
            pi_value=1.0,
            estimate_pi=False,
            perform_qc=False,
            center_genotypes=True
        )

        model.set_mcmc_options(
            chain_length=60,
            burn_in=10,
            thinning=2,
            seed=42
        )

        print(f"Running BayesC0 integration test model. Results will be captured in {self.results_capture_file}")
        results = None
        run_error = None
        try:
            # The model.run() itself might not directly return detailed DataFrames in the API for GenoStockModel
            # It would populate self.model._mme.posterior_means and self.model._mme.posterior_samples
            # For this test, we'll access them after the run.
            model.run(output_folder=str(self.base_test_dir / "bayes_c0_mcmc_output")) # Dummy output folder
        except Exception as e:
            run_error = e
            print(f"model.run() raised an exception: {e}") # Print error to aid debugging if run fails

        # --- Output Capturing and Sanity Checks ---
        output_lines = ["Test Script: test_bayesc0_run_simple_model"]
        if run_error:
            output_lines.append(f"RUN_FAILED: {run_error}")
        else:
            output_lines.append("RUN_COMPLETED_SUCCESSFULLY")
            self.assertTrue(model._is_prepared)
            self.assertTrue(model._analysis_run)

            # Access results from model._mme (which is an MME_py instance)
            mme_results = model._mme.posterior_means # This is a Dict[str, pd.DataFrame or value]

            res_var_df = mme_results.get("residual_variance")
            sigma_e2_mean = res_var_df["Estimate"].iloc[0] if isinstance(res_var_df, pd.DataFrame) and not res_var_df.empty else None
            output_lines.append(f"sigma_e2_mean={sigma_e2_mean if sigma_e2_mean is not None else 'N/A'}")
            if sigma_e2_mean is not None: self.assertTrue(sigma_e2_mean > 0)

            sigma_g2_mean = None
            bayes_c0_gc_name = "chip_data"
            key_marker_var = next((k for k in mme_results if bayes_c0_gc_name in k and ("variance" in k or "variances" in k)), None)
            if key_marker_var and mme_results.get(key_marker_var) is not None:
                marker_var_df = mme_results[key_marker_var]
                if isinstance(marker_var_df, pd.DataFrame) and not marker_var_df.empty:
                    sigma_g2_mean = marker_var_df["Estimate"].iloc[0]
            output_lines.append(f"sigma_g2_mean={sigma_g2_mean if sigma_g2_mean is not None else 'N/A'}")
            if sigma_g2_mean is not None: self.assertTrue(sigma_g2_mean > 0)

            loc_params_df = mme_results.get("location_parameters")
            if loc_params_df is not None and not loc_params_df.empty:
                output_lines.append("location_parameters_present=True")
                intercept_val = loc_params_df[loc_params_df["parameter"].str.contains("intercept", case=False)]["mean"].iloc[0]
                sex_effect_val = loc_params_df[loc_params_df["parameter"].str.contains("sex", case=False)]["mean"].iloc[0] # Simplified
                output_lines.append(f"intercept_mean={intercept_val:.4f}")
                output_lines.append(f"sex_effect_mean={sex_effect_val:.4f}") # Example, actual name may vary
            else:
                output_lines.append("location_parameters_present=False")

            marker_effects_key = f"marker_effects_{bayes_c0_gc_name}_{model._mme.get_trait_name(0)}" # Construct key as expected
            marker_effects_df = mme_results.get(marker_effects_key)
            if marker_effects_df is not None and not marker_effects_df.empty:
                output_lines.append("marker_effects_present=True")
                output_lines.append(f"marker_effect_1_mean={marker_effects_df['Estimate'].iloc[0]:.6f}")
                output_lines.append(f"marker_effect_last_mean={marker_effects_df['Estimate'].iloc[-1]:.6f}")
                self.assertEqual(len(marker_effects_df), self.n_markers)
                if len(marker_effects_df["Estimate"]) > 1:
                     self.assertTrue(marker_effects_df["Estimate"].std() > 1e-9)
            else:
                output_lines.append("marker_effects_present=False")

        with open(self.results_capture_file, "w") as f:
            f.write("\n".join(output_lines))

        if run_error: # Re-raise error if model run failed, after writing to file
            raise run_error


if __name__ == '__main__':
    unittest.main()
