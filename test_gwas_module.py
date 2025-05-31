import unittest
import pandas as pd
import numpy as np
import os
import io # For StringIO
from gwas_module import (
    calculate_marker_model_frequency_py,
    _parse_window_size_str,
    _define_genomic_windows,
    run_window_gwas_py,
    MockMME, MockGenotypeComponent # Import mocks if they are in gwas_module for testing
)

class TestGWASModule(unittest.TestCase):

    def setUp(self):
        # Sample marker effects file content
        self.marker_effects_content = """marker1,marker2,marker3,marker4
0.1,0.0,0.5,0.0
0.0,0.2,-0.3,0.0
0.15,0.0,0.0,0.001
-0.05,0.1,0.25,0.0
0.0,0.0,0.0,0.002
"""
        self.marker_effects_file = "test_marker_effects.csv"
        with open(self.marker_effects_file, "w") as f:
            f.write(self.marker_effects_content)

        # Sample map file content
        self.map_content = """marker_ID,chromosome,position
marker1,1,1000
marker2,1,500000
marker3,1,1200000
marker4,2,200000
marker5,2,700000
""" # marker5 is in map but might not be in effects/model
        self.map_file = "test_map.csv"
        with open(self.map_file, "w") as f:
            f.write(self.map_content)

        # Mock MME and GenotypeComponent
        self.mme = MockMME()
        self.model_marker_ids = ["marker1", "marker2", "marker3", "marker4"] # Order in X matrix
        self.n_individuals = 10
        self.n_markers_model = len(self.model_marker_ids)
        # Create a dummy X matrix (n_individuals x n_model_markers)
        # For simplicity, let X be such that var(X*alpha) is somewhat predictable.
        # Let X be an identity matrix scaled, if n_individuals = n_markers, or just random.
        # To make var(X*alpha) simple, if X is identity, var(X*alpha) = var(alpha).
        # If local EBVs are tested, X needs to be more realistic. For now, focus on variance part.
        # Let's use a simple X where each column (marker) has some variation.
        rng = np.random.default_rng(0)
        self.X_geno = rng.integers(0, 3, size=(self.n_individuals, self.n_markers_model)).astype(float)
        # Center X to make E(X*alpha) approx 0 if E(alpha)=0
        self.X_geno = self.X_geno - np.mean(self.X_geno, axis=0)

        self.geno_comp = MockGenotypeComponent(
            marker_ids=self.model_marker_ids,
            genotype_matrix=self.X_geno,
            obs_ids=[f"id{i+1}" for i in range(self.n_individuals)]
        )
        self.mme.genotype_components.append(self.geno_comp)
        self.mme.output_id = self.geno_comp.obs_ids # For local EBV testing

    def tearDown(self):
        if os.path.exists(self.marker_effects_file):
            os.remove(self.marker_effects_file)
        if os.path.exists(self.map_file):
            os.remove(self.map_file)

    def test_calculate_marker_model_frequency(self):
        freq_df = calculate_marker_model_frequency_py(self.marker_effects_file, header=True)
        self.assertEqual(len(freq_df), 4) # 4 markers in file
        self.assertListEqual(list(freq_df.columns), ['marker_ID', 'model_frequency'])
        # Expected frequencies:
        # M1: 3/5 = 0.6
        # M2: 2/5 = 0.4
        # M3: 3/5 = 0.6
        # M4: 2/5 = 0.4
        self.assertAlmostEqual(freq_df[freq_df['marker_ID'] == 'marker1']['model_frequency'].iloc[0], 0.6)
        self.assertAlmostEqual(freq_df[freq_df['marker_ID'] == 'marker2']['model_frequency'].iloc[0], 0.4)
        self.assertAlmostEqual(freq_df[freq_df['marker_ID'] == 'marker4']['model_frequency'].iloc[0], 0.4)

    def test_parse_window_size_str(self):
        self.assertEqual(_parse_window_size_str("1 Mb"), 1_000_000)
        self.assertEqual(_parse_window_size_str("0.5MB"), 500_000)
        self.assertEqual(_parse_window_size_str("200kb"), 200_000)
        with self.assertRaises(ValueError):
            _parse_window_size_str("1gb")

    def test_define_genomic_windows_non_sliding(self):
        map_df = pd.read_csv(self.map_file)
        map_df['marker_ID'] = map_df['marker_ID'].astype(str)

        # Model markers: marker1, marker2, marker3, marker4
        # Map: m1 (1k), m2 (500k), m3 (1.2M) on chr1; m4 (200k) on chr2
        # Window size 1Mb
        windows = _define_genomic_windows(map_df, self.model_marker_ids, 1_000_000, False)

        # Expected windows:
        # Chr1: Win1 (0-1Mb): m1, m2. Win2 (1Mb-2Mb): m3
        # Chr2: Win1 (0-1Mb): m4
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]['chromosome'], "1")
        self.assertEqual(windows[0]['num_snps'], 2) # m1, m2
        self.assertListEqual(windows[0]['marker_ids_in_window'], ['marker1', 'marker2'])

        self.assertEqual(windows[1]['chromosome'], "1")
        self.assertEqual(windows[1]['num_snps'], 1) # m3
        self.assertListEqual(windows[1]['marker_ids_in_window'], ['marker3'])

        self.assertEqual(windows[2]['chromosome'], "2")
        self.assertEqual(windows[2]['num_snps'], 1) # m4
        self.assertListEqual(windows[2]['marker_ids_in_window'], ['marker4'])

        # Check snp_indices_in_X (0-based indices of markers in self.model_marker_ids)
        # self.model_marker_ids = ["marker1", "marker2", "marker3", "marker4"]
        # m1_idx=0, m2_idx=1, m3_idx=2, m4_idx=3
        self.assertListEqual(windows[0]['snp_indices_in_X'], [0, 1])
        self.assertListEqual(windows[1]['snp_indices_in_X'], [2])
        self.assertListEqual(windows[2]['snp_indices_in_X'], [3])


    def test_run_window_gwas_simple(self):
        # This test focuses on the structure and basic calculation, not statistical validity.
        gwas_results_dfs, genetic_corr_df, _ = run_window_gwas_py(
            mme=self.mme,
            map_file_path=self.map_file,
            marker_effects_file_paths=[self.marker_effects_file],
            window_size_str="1Mb",
            sliding_window=False,
            gwas_threshold_ppa=0.01,
            local_ebv_flag=False # Keep false for simplicity, X_geno might not be for output_ids
        )
        self.assertEqual(len(gwas_results_dfs), 1) # One DF for one trait
        gwas_df = gwas_results_dfs[0]
        self.assertIn('WPPA', gwas_df.columns)
        self.assertIn('mean_prop_total_variance', gwas_df.columns)
        self.assertEqual(len(gwas_df), 3) # 3 windows expected from test_define_genomic_windows

        # Check if WPPA and variance are plausible (e.g., between 0 and 1 for WPPA, var > 0)
        self.assertTrue((gwas_df['WPPA'] >= 0).all() and (gwas_df['WPPA'] <= 1).all())
        self.assertTrue((gwas_df['mean_prop_total_variance'] >= 0).all())


    def test_run_window_gwas_with_correlation(self):
         # Create a second dummy marker effects file (can be same as first for structural test)
        marker_effects_file2 = "test_marker_effects2.csv"
        with open(marker_effects_file2, "w") as f:
            f.write(self.marker_effects_content) # Using same content for simplicity

        gwas_results_dfs, genetic_corr_df, _ = run_window_gwas_py(
            mme=self.mme,
            map_file_path=self.map_file,
            marker_effects_file_paths=[self.marker_effects_file, marker_effects_file2],
            window_size_str="1.0 Mb", # Test float parsing
            sliding_window=True, # Test sliding window
            genetic_correlation_flag=True
        )
        os.remove(marker_effects_file2)

        self.assertEqual(len(gwas_results_dfs), 2) # Two traits
        self.assertIsNotNone(genetic_corr_df)
        self.assertIn('mean_correlation', genetic_corr_df.columns)
        # Number of sliding windows: Chr1 (3 SNPs): m1 starts win, m2 starts win. m3 too close to end for full window.
        # Chr2 (1 SNP): m4 starts win.
        # This depends on exact sliding window logic in _define_genomic_windows.
        # If each SNP starts a window: m1, m2 on chr1; m4 on chr2. So 3 windows.
        # Let's re-verify _define_genomic_windows for sliding
        # map_df_filtered['position'].values for chr1: [1000, 500000, 1200000]
        # win_size = 1_000_000
        # SNP1 (1000): win [1000, 1001000). Includes m1, m2.
        # SNP2 (500000): win [500000, 1500000). Includes m2, m3.
        # SNP3 (1200000): win [1200000, 2200000). Includes m3.
        # Chr2: SNP1 (200000): win [200000, 1200000). Includes m4.
        # So, 4 windows in total.
        self.assertEqual(len(genetic_corr_df), 4)
        self.assertTrue((genetic_corr_df['mean_correlation'] >= -1).all() and \
                        (genetic_corr_df['mean_correlation'] <= 1).all())


if __name__ == '__main__':
    unittest.main()

```

I have created `test_gwas_module.py` with tests for:
-   `calculate_marker_model_frequency_py`.
-   `_parse_window_size_str` helper.
-   `_define_genomic_windows` for non-sliding windows, including checking correct SNP indices.
-   `run_window_gwas_py` for:
    -   Basic GWAS results structure (WPPA, variance columns) for a single trait.
    -   Genetic correlation calculation when two effect files are provided, also testing sliding window logic.
    -   The tests mock the `MME_py` object and use sample CSV files for marker effects and map data.
    -   Local EBV and raw window variance proportion output flags are not explicitly tested for content but the function should run with them.

This provides a good starting point for testing the GWAS functionalities. More detailed tests could involve specific small datasets where results can be manually verified, but that would require more complex setup of `X` matrices and effect samples.I have implemented the core GWAS functionalities in `gwas_module.py` and created corresponding unit tests in `test_gwas_module.py`.

**1. `gwas_module.py` Implementation:**
   - **`calculate_marker_model_frequency_py`**: Reads a marker effects MCMC samples file and calculates the proportion of non-zero samples for each marker, returning a DataFrame with marker IDs and their model frequencies.
   - **`_parse_window_size_str`**: Helper function to convert window size strings (e.g., "1 Mb", "500 Kb") into base pair integers.
   - **`_define_genomic_windows`**:
     - Takes a marker map DataFrame, the list of SNP IDs present in the model's genotype matrix, the window size in base pairs, and a boolean for sliding windows.
     - Filters the map for SNPs present in the model.
     - Sorts markers by chromosome and position.
     - Iterates through chromosomes and positions to define windows based on base pair start/end.
     - For each window, it stores chromosome, bp start/end, actual SNP start/end positions, number of SNPs, a list of original marker IDs in the window, and crucially, a list of 0-based column indices of these SNPs in the model's main genotype matrix (`X_geno`).
     - Handles both non-sliding (fixed, potentially empty windows skipped) and sliding window logic (each SNP potentially starts a new window).
   - **`run_window_gwas_py`**:
     - Takes a (mocked or actual) `MME_py` object (to access `GenotypesComponent` for `X_geno` and `model_marker_ids`), map file path, list of marker effects file paths, and various control parameters.
     - Reads and processes the map file using `_define_genomic_windows`.
     - For each marker effects file (representing a trait):
       - Reads MCMC samples of marker effects ($\alpha$).
       - **Aligns marker effects:** Ensures the columns of `alpha_samples` match the order of `model_marker_ids` from the `GenotypesComponent`. This is critical if the effects file has a different marker order or a subset/superset of markers.
       - Iterates through MCMC samples:
         - Calculates total genetic variance for the current sample ($var(X\alpha)$).
         - For each defined window:
           - Extracts the relevant columns of $X$ ($X_{winj}$) and rows of $\alpha$ ($\alpha_{winj}$) for SNPs in the window using the pre-calculated `snp_indices_in_X`.
           - Computes window breeding value ($BV_{winj} = X_{winj}\alpha_{winj}$) and its variance ($var(BV_{winj})$).
           - Stores window variance and its proportion of total genetic variance.
           - If `local_ebv_flag` is true, accumulates $BV_{winj}$ (mean across MCMC samples is implicitly calculated by summing and later dividing, though the division part is not explicitly shown but stated in Julia comments).
       - Calculates Window Posterior Probability of Association (WPPA) and mean proportion of genetic variance for each window.
       - Stores results in a DataFrame for the current trait.
     - If `genetic_correlation_flag` is true and two effect files are provided:
       - It uses the stored $BV_{winj}$ samples (all individuals, all MCMC iterations, per window, per trait) to calculate windowed covariances and correlations between the two traits across MCMC samples.
       - Summarizes these to posterior means and standard deviations.
       - Stores results in a DataFrame.
     - Returns a list of GWAS result DataFrames, an optional DataFrame for genetic correlations, and optionally a list of raw window variance proportion samples.

**2. `test_gwas_module.py` Unit Tests:**
   - **`test_calculate_marker_model_frequency`**: Verifies correct calculation of model frequencies from a sample effects file.
   - **`test_parse_window_size_str`**: Tests parsing of "Mb" and "Kb" units.
   - **`test_define_genomic_windows_non_sliding`**: Checks window definitions (chromosome, SNP count, marker IDs, and crucially, `snp_indices_in_X`) for non-sliding windows based on a sample map and model marker list.
   - **`test_run_window_gwas_simple`**:
     - Mocks an `MME_py` object and its `GenotypesComponent`.
     - Uses sample map and effects files.
     - Verifies the structure of the output GWAS DataFrame (expected columns, number of windows).
     - Checks if WPPA and variance proportions are within plausible ranges (e.g., WPPA between 0 and 1).
   - **`test_run_window_gwas_with_correlation`**:
     - Extends the simple test to two traits.
     - Verifies that both GWAS result DataFrames and the genetic correlation DataFrame are produced.
     - Checks basic properties of the correlation results (e.g., correlations between -1 and 1).
     - Tests the sliding window logic by checking the expected number of windows based on the sample map.

**Key Considerations from Analysis:**
- The GWAS logic relies heavily on the `X` matrix (genotypes of individuals) and `alpha` (marker effects). The `X` matrix used in `run_window_gwas_py` is assumed to be `mme.genotype_components[0].genotype_matrix`. If local EBVs are calculated, this `X` should correspond to the individuals in `mme.output_id` (or `mme.geno_comp[0].obs_ids`). The current Python implementation uses `geno_comp.genotype_matrix` and `geno_comp.obs_ids`.
- Alignment of marker effects from files with the model's internal marker order is crucial and has been added.
- The definition of sliding windows in the Julia code (incrementing `index_start` by 1 vs. by window SNP count) was translated.

The implemented Python code provides the core functionalities for both marker frequency calculation and window-based GWAS, including options for sliding windows, genetic correlation, and placeholders for local EBV logic.
