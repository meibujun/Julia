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

