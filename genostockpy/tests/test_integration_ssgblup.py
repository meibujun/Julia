import unittest
import pandas as pd
import numpy as np
import os
import shutil
import pathlib

from genostockpy.api import GenoStockModel

class TestIntegrationSSGBLUP(unittest.TestCase):

    def setUp(self):
        self.base_test_dir = pathlib.Path("_test_integration_ssgblup_data")
        self.ssgblup_data_dir = self.base_test_dir / "data" / "ssgblup_example"
        os.makedirs(self.ssgblup_data_dir, exist_ok=True)

        self.results_capture_file = pathlib.Path("/tmp/ssgblup_results.txt")
        if self.results_capture_file.exists():
            os.remove(self.results_capture_file)

        self.n_ind = 20
        self.n_markers = 50
        self._create_dummy_ssgblup_data()

        self.pheno_file = self.ssgblup_data_dir / "phenotypes.csv"
        self.ped_file = self.ssgblup_data_dir / "pedigree.csv"
        self.geno_file = self.ssgblup_data_dir / "genotypes.csv"


    def _create_dummy_ssgblup_data(self):
        with open(self.pheno_file, "w") as f:
            f.write("ID,y,group\n")
            for i in range(1, self.n_ind + 1):
                group_val = (i % 2) + 1
                random_int_part = np.random.randint(0,5)
                y_val = 20 + group_val * 2 + random_int_part
                f.write(f"animal{i},{float(y_val)},Group{group_val}\n")

        with open(self.ped_file, "w") as f:
            f.write("ID,Sire,Dam\n")
            f.write("animal1,0,0\n")
            f.write("animal2,0,0\n")
            f.write("animal3,0,0\n")
            f.write("animal4,0,0\n")
            f.write("animal5,animal1,animal2\n")
            f.write("animal6,animal3,animal4\n")
            f.write("animal7,animal1,animal3\n")
            f.write("animal8,animal4,animal5\n")
            for i in range(9, self.n_ind + 1):
                 sire_idx = np.random.randint(1,i) if i > 1 else 0
                 dam_idx = np.random.randint(1,i) if i > 1 else 0
                 sire = f"animal{sire_idx}" if sire_idx > 0 else "0"
                 dam = f"animal{dam_idx}" if dam_idx > 0 else "0"
                 if sire == dam and sire != "0": dam = "0"
                 f.write(f"animal{i},{sire},{dam}\n")

        genotyped_ids=["animal3", "animal4", "animal6", "animal8"] + [f"animal{i}" for i in range(9,19+1)]
        with open(self.geno_file, "w") as f:
            header = "ID," + ",".join([f"M{j+1}" for j in range(self.n_markers)])
            f.write(header + "\n")
            for id_val in genotyped_ids:
                 if int(id_val.replace("animal","")) <= self.n_ind :
                    line = f"{id_val}," + ",".join(map(str, np.random.randint(0, 3, size=self.n_markers)))
                    f.write(line + "\n")

    def tearDown(self):
        if self.base_test_dir.exists():
            shutil.rmtree(self.base_test_dir)
        # if self.results_capture_file.exists():
        #     os.remove(self.results_capture_file)

    def test_ssgblup_run_simple_model(self):
        model = GenoStockModel(model_name="SSGBLUP_IntegrationTest")
        model.set_model_equation(
            equation="y = intercept + group + animal_id",
            trait_types={"y": "continuous"}
        )
        model.load_phenotypes(
            data=str(self.pheno_file),
            id_column="ID",
        )
        model.load_pedigree(
            file_path=str(self.ped_file),
            header=True, separator=',', missing_strings=["0"]
        )
        model.add_genotypes(
            name="geno_for_ssgblup",
            data_source=str(self.geno_file),
            method="GBLUP",
            genetic_variance_value=1.5,
            df_prior=5.0,
            perform_qc=False,
            center_genotypes=True
        )
        model.add_random_effect(
            effect_name="animal_id",
            use_pedigree=True,
        )
        model.set_mcmc_options(
            chain_length=60,
            burn_in=10,
            thinning=2,
            seed=123,
            single_step_analysis=True
        )

        print(f"Running SSGBLUP integration test model. Results will be captured in {self.results_capture_file}")
        results = None
        run_error = None
        try:
            results = model.run(output_folder=str(self.base_test_dir / "ssgblup_mcmc_output"))
        except Exception as e:
            run_error = e
            print(f"model.run() for SSGBLUP raised an exception: {e}")


        output_lines = ["SSGBLUP Integration Test Output Summary:"]
        if run_error:
            output_lines.append(f"RUN_FAILED: {run_error}")
        else:
            output_lines.append("RUN_COMPLETED_SUCCESSFULLY")
            self.assertTrue(model._is_prepared)
            self.assertTrue(model._analysis_run)
            self.assertIsNotNone(results)

            mme_results = model._mme.posterior_means

            res_var_df = mme_results.get("residual_variance")
            sigma_e2_mean = res_var_df["Estimate"].iloc[0] if isinstance(res_var_df, pd.DataFrame) and not res_var_df.empty else None
            output_lines.append(f"  Posterior Mean Residual Variance (sigma_e^2): {sigma_e2_mean if sigma_e2_mean is not None else 'N/A':.4f}")
            if sigma_e2_mean is not None: self.assertTrue(sigma_e2_mean > 0)

            animal_var_key = None
            if isinstance(mme_results, dict): # Check if mme_results is dict
                possible_keys = [f"VC_animal_id", "polygenic_effects_covariance_matrix"]
                if model._mme.genotype_components:
                    possible_keys.append(f"genetic_variance_{model._mme.genotype_components[0].name}")
                for key_attempt in possible_keys:
                    if key_attempt in mme_results: animal_var_key = key_attempt; break
                if not animal_var_key:
                    for key in mme_results.keys():
                        if ("polygenic" in key or "animal_id" in key or (model._mme.genotype_components and model._mme.genotype_components[0].name in key)) and \
                           ("variance" in key or "covariance" in key) and not ("effects_variances" in key and "marker" in key):
                            animal_var_key = key; break

            sigma_a2_mean = None
            if animal_var_key and mme_results.get(animal_var_key) is not None:
                animal_var_df = mme_results[animal_var_key]
                if isinstance(animal_var_df, pd.DataFrame) and not animal_var_df.empty:
                    sigma_a2_mean = animal_var_df["Estimate"].iloc[0]
            output_lines.append(f"  Posterior Mean Genetic Variance (sigma_a^2 for {animal_var_key or 'animal_id'}): {sigma_a2_mean if sigma_a2_mean is not None else 'N/A':.4f}")
            if sigma_a2_mean is not None: self.assertTrue(sigma_a2_mean > 0)


            loc_params_df = mme_results.get("location_parameters")
            if loc_params_df is not None and not loc_params_df.empty:
                output_lines.append("\n  Posterior Means for Location Parameters (Fixed Effects):")
                fixed_effects_sub_df = loc_params_df[loc_params_df['parameter'].str.contains("intercept|group", case=False, na=False)]
                output_lines.append(fixed_effects_sub_df.to_string())
                if not fixed_effects_sub_df.empty : # Check only if DF is not empty
                    self.assertTrue(any("intercept" in p_name.lower() for p_name in loc_params_df["parameter"]))
                    self.assertTrue(any("group" in p_name.lower() for p_name in loc_params_df["parameter"]))
            else: output_lines.append("  Location parameters not found/empty.")

            ebv_df = model.get_ebv()
            if ebv_df is not None and not ebv_df.empty:
                output_lines.append("\n  Sample EBVs (first 5):")
                output_lines.append(ebv_df.head().to_string())
                self.assertEqual(len(ebv_df), self.n_ind * model._mme.n_models)
                self.assertTrue(np.all(np.isfinite(ebv_df["EBV"])))
            else:
                output_lines.append("  No EBVs retrieved or EBV DataFrame is empty.")

        with open(self.results_capture_file, "w") as f:
            f.write("\n".join(output_lines))

        if run_error: raise run_error # Re-raise error after writing to file

if __name__ == '__main__':
    import pathlib
    unittest.main()
