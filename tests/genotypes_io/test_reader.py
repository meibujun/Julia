import unittest
import pandas as pd
import numpy as np
import os

from pyjwas.genotypes_io import get_genotypes_data
from pyjwas.core import GenotypesData

class TestGenotypeReader(unittest.TestCase):

    def _create_dummy_geno_file(self, filename="test_genos.csv", content=None, separator=','):
        if content is None:
            content = f"""ObsID{separator}M1{separator}M2{separator}M3{separator}M4
id1{separator}0{separator}1{separator}2{separator}9
id2{separator}1{separator}1{separator}0{separator}1
id3{separator}2{separator}0{separator}1{separator}0
id4{separator}1{separator}2{separator}missing{separator}1
""" # 'missing' will be NaN, '9' will be 9.0
        with open(filename, "w") as f:
            f.write(content.strip())
        return filename

    def tearDown(self):
        if os.path.exists("test_genos.csv"): os.remove("test_genos.csv")
        if os.path.exists("IDs_for_individuals_with_genotypes.txt"): # Written by get_genotypes_data
            os.remove("IDs_for_individuals_with_genotypes.txt")


    def test_read_from_csv_basic(self):
        filepath = self._create_dummy_geno_file()
        geno_data = get_genotypes_data(
            filepath,
            file_separator=',',
            file_has_header=True, # M1, M2.. are marker IDs
            quality_control=False, # Turn off QC for direct check of reading
            center_data=False,
            missing_value_code=9.0, # Numeric missing code
            genotype_data_name="test_csv"
        )
        self.assertIsInstance(geno_data, GenotypesData)
        self.assertEqual(geno_data.name, "test_csv")
        self.assertEqual(geno_data.n_obs, 4)
        self.assertEqual(geno_data.n_markers, 4) # M1, M2, M3, M4
        self.assertEqual(geno_data.obs_ids, ["id1", "id2", "id3", "id4"])
        self.assertEqual(geno_data.marker_ids, ["M1", "M2", "M3", "M4"])

        # Expected raw matrix (missing 'missing' becomes NaN, '9' is 9.0)
        # After fillna(missing_value_code) in reader for pd.NA, 'missing' becomes 9.0
        expected_matrix = np.array([
            [0., 1., 2., 9.], # 9 remains 9
            [1., 1., 0., 1.],
            [2., 0., 1., 0.],
            [1., 2., 9., 1.]  # 'missing' -> NaN -> fillna(9.0) -> 9.0
        ])
        np.testing.assert_array_almost_equal(geno_data.genotypes, expected_matrix)


    def test_read_from_dataframe(self):
        data = {
            'Animal': ['cow1', 'cow2', 'cow3'],
            'snp1': [0,1,2],
            'snp2': [1,2,0],
            'snp3': [2,0,np.nan] # Use np.nan for pandas native missing
        }
        df = pd.DataFrame(data)
        geno_data = get_genotypes_data(
            df,
            obs_ids_col_name_or_index='Animal',
            quality_control=True, # Test QC with NaN
            missing_value_code=np.nan, # Tell QC that NaN is the code
            maf_threshold=0.0, # No MAF filtering for this simple test
            center_data=True
        )
        self.assertEqual(geno_data.n_obs, 3)
        self.assertEqual(geno_data.n_markers, 3) # All markers should pass if MAF=0
        self.assertEqual(geno_data.obs_ids, ['cow1', 'cow2', 'cow3'])
        self.assertEqual(geno_data.marker_ids, ['snp1', 'snp2', 'snp3'])

        # Check centering (sum of columns should be close to 0)
        self.assertTrue(geno_data.is_centered)
        np.testing.assert_allclose(np.sum(geno_data.genotypes, axis=0), np.zeros(3), atol=1e-9)

        # Check imputation: snp3 had a NaN. Mean of (2,0) is 1. So NaN becomes 1.
        # Original snp3: [2, 0, NaN] -> [2,0,1] before centering.
        # Means: snp1=1, snp2=1, snp3=1
        # Centered snp3: [2-1, 0-1, 1-1] = [1, -1, 0]
        self.assertAlmostEqual(geno_data.genotypes[2,2], 0.0) # (1-1)


    def test_read_from_numpy_array(self):
        arr = np.array([[0,1,2],[1,2,9]], dtype=float) # 9 as missing
        obs_ids = ["obsA", "obsB"]
        mrk_ids = ["mkrX", "mkrY", "mkrZ"]
        geno_data = get_genotypes_data(
            arr,
            obs_ids_list=obs_ids,
            marker_ids_list=mrk_ids,
            quality_control=True, missing_value_code=9.0, maf_threshold=0.0,
            center_data=False # Test without centering
        )
        self.assertEqual(geno_data.obs_ids, obs_ids)
        self.assertEqual(geno_data.marker_ids, mrk_ids)
        # Missing 9 in arr[1,2] should be imputed. Mean of arr[:,2] (2,9) non-missing is 2.
        # So arr[1,2] becomes 2.
        expected_processed = np.array([[0.,1.,2.],[1.,2.,2.]])
        np.testing.assert_array_almost_equal(geno_data.genotypes, expected_processed)
        self.assertFalse(geno_data.is_centered)

    def test_qc_maf_filter(self):
        # M1: p=0.5 (MAF=0.5), M2: p=0.05 (MAF=0.05), M3: p=0 (fixed), M4: p=0.98 (MAF=0.02)
        content = """ID,M1,M2,M3,M4
id1,1,0,0,2
id2,1,0,0,2
id3,1,1,0,2
id4,1,0,0,2
id5,1,0,0,2
id6,1,0,0,2
id7,1,0,0,2
id8,1,0,0,2
id9,1,0,0,2
id10,1,0,0,2
""" # M2 has 1 '1' out of 10 '0's -> p=0.1/2=0.05. M4 has all 2s -> p=1 (MAF=0).
        # Let's fix M4 to have some variation for MAF=0.02 test
        # M4: two 1's, eight 2's. Mean = (2*1 + 8*2)/10 = 1.8. p = 0.9. MAF = 0.1. (Not 0.02)
        # To get MAF 0.02 (p=0.02 or p=0.98). e.g. p=0.02: 2*p*N = 0.02*2*10 = 0.4 ones. (approx)
        # Say for 50 inds, p=0.02 means 1 '1' and 49 '0's. Mean = 1/50 = 0.02. p=0.01. MAF=0.01.
        # Let's use a clearer example for MAF.
        # M1: 5 zeros, 5 twos. Mean = 1. p = 0.5. MAF = 0.5. (Keep)
        # M2: 1 two, 9 zeros. Mean = 0.2. p = 0.1. MAF = 0.1. (Keep if MAF_thresh <= 0.1)
        # M3: 10 zeros. Mean = 0. p = 0. MAF = 0. (Remove)
        # M4: 1 zero, 9 twos. Mean = 1.8. p = 0.9. MAF = 0.1. (Keep if MAF_thresh <= 0.1)
        # M5: 1 one, 9 zeros. Mean = 0.1. p = 0.05. MAF = 0.05. (Keep if MAF_thresh <= 0.05)
        content_maf = """ID,M1,M2,M3,M4,M5
i1,0,0,0,2,0
i2,0,0,0,2,0
i3,0,0,0,2,0
i4,0,0,0,2,0
i5,0,2,0,2,1
i6,2,0,0,0,0
i7,2,0,0,2,0
i8,2,0,0,2,0
i9,2,0,0,2,0
i10,2,2,0,2,0
"""
        # M1: 5x0, 5x2. Mean=1. p=0.5. MAF=0.5.
        # M2: 2x2, 8x0. Mean=0.4. p=0.2. MAF=0.2.
        # M3: 10x0. Mean=0. p=0. MAF=0. (Removed)
        # M4: 1x0, 9x2. Mean=1.8. p=0.9. MAF=0.1.
        # M5: 1x1, 9x0. Mean=0.1. p=0.05. MAF=0.05.
        filepath = self._create_dummy_geno_file(content=content_maf)
        geno_data = get_genotypes_data(filepath, maf_threshold=0.08, center_data=False, quality_control=True)

        # Expected to keep M1 (MAF 0.5), M2 (MAF 0.2), M4 (MAF 0.1).
        # Remove M3 (MAF 0), M5 (MAF 0.05 < 0.08).
        self.assertEqual(geno_data.n_markers, 3)
        self.assertEqual(set(geno_data.marker_ids), set(["M1", "M2", "M4"]))

    def test_gblup_grm_calculation(self):
        # Simple Z: [[0,2],[2,0]] -> centered Zc: [[-1,1],[1,-1]] (means are 1,1)
        # p = [0.5, 0.5]. 2pq = [0.5, 0.5] per marker. sqrt(2pq) = [0.707, 0.707]
        # Z_std_col0 = [-1/0.707, 1/0.707] = [-1.414, 1.414]
        # Z_std_col1 = [1/0.707, -1/0.707] = [1.414, -1.414]
        # Z_std = [[-1.414, 1.414], [1.414, -1.414]]
        # ZZ' = [[4, -4],[-4, 4]].  ZZ'/n_markers = [[2,-2],[-2,2]]
        arr = np.array([[0.,2.],[2.,0.]])
        geno_data = get_genotypes_data(arr, method="GBLUP", center_data=True, quality_control=False) # QC false to keep all markers
        self.assertTrue(geno_data.is_grm)
        self.assertEqual(geno_data.genotypes.shape, (2,2))
        # Calculation:
        # Means: [1,1]. p=[0.5,0.5]. 2p(1-p) = [0.5, 0.5]. sqrt(denom) = [0.7071, 0.7071]
        # Z_centered = [[-1,1],[1,-1]]
        # Z_std = [[-1.4142, 1.4142],[1.4142, -1.4142]]
        # Z_std @ Z_std.T = [[4,-4],[-4,4]]
        # GRM = ZZ'/n_markers = [[2,-2],[-2,2]]
        expected_grm = np.array([[2.,-2.],[-2.,2.]])
        np.testing.assert_allclose(geno_data.genotypes, expected_grm, atol=1e-5)

    def test_gblup_input_is_grm(self):
        grm = np.array([[1.0, 0.5],[0.5, 1.0]])
        geno_data = get_genotypes_data(grm, method="GBLUP")
        self.assertTrue(geno_data.is_grm)
        np.testing.assert_array_equal(geno_data.genotypes, grm) # Should use it as is
        self.assertFalse(geno_data.is_centered) # Centering not applied to pre-computed GRM
        self.assertIsNone(geno_data.allele_freqs)


if __name__ == '__main__':
    unittest.main()
```
