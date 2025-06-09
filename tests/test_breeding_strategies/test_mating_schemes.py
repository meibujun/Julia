# tests/test_breeding_strategies/test_mating_schemes.py

import unittest
import pandas as pd
import numpy as np
import random

# Adjust path to import from the root of the project
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.breeding_strategies.mating_schemes import (
    random_mating,
    calculate_progeny_inbreeding,
    generate_mating_list_with_inbreeding
)
# For creating a sample NRM for testing inbreeding calculations:
# from sheep_breeding_genomics.genetic_evaluation.relationship_matrix import calculate_nrm
# from sheep_breeding_genomics.data_management.data_structures import PedigreeData
# For simplicity in this test file, we'll often create a placeholder NRM DataFrame directly.

class TestMatingSchemes(unittest.TestCase):

    def setUp(self):
        self.sires_df = pd.DataFrame({'AnimalID': ['S1', 'S2', 'S3']})
        self.dams_df = pd.DataFrame({'AnimalID': ['D1', 'D2', 'D3', 'D4', 'D5']})

        # Sample NRM for testing inbreeding
        self.animal_ids_nrm = ['S1', 'S2', 'S3', 'D1', 'D2', 'D3', 'D4', 'D5', 'X1']
        nrm_size = len(self.animal_ids_nrm)
        self.nrm_values = np.identity(nrm_size)
        # Make S1 and D1 related (e.g. full sibs with F=0.25, so relationship = 0.5 due to A_ii = 1+F_i for NRM diagonal,
        # but here we mean off-diagonal relationship for progeny inbreeding calc is A_sd)
        # Let A(S1,D1) = 0.5 (e.g. they are full-sibs themselves, not their progeny's F)
        # Or, if S1 and D1 are parents of an inbred individual, their relationship is what matters.
        # Let's set a known relationship for S1-D1.
        s1_idx = self.animal_ids_nrm.index('S1')
        d1_idx = self.animal_ids_nrm.index('D1')
        self.nrm_values[s1_idx, d1_idx] = self.nrm_values[d1_idx, s1_idx] = 0.25 # Arbitrary relationship for test

        s2_idx = self.animal_ids_nrm.index('S2')
        d2_idx = self.animal_ids_nrm.index('D2')
        self.nrm_values[s2_idx, d2_idx] = self.nrm_values[d2_idx, s2_idx] = 0.0 # Unrelated

        self.nrm_df = pd.DataFrame(self.nrm_values, index=self.animal_ids_nrm, columns=self.animal_ids_nrm)

    def test_random_mating_basic(self):
        matings = random_mating(self.sires_df, self.dams_df, sire_id_col='AnimalID', dam_id_col='AnimalID')
        self.assertTrue(len(matings) > 0) # Probabilistic, but with 3 sires, 5 dams, should get matings
        for sire, dam, prog_num in matings:
            self.assertIn(sire, self.sires_df['AnimalID'].tolist())
            self.assertIn(dam, self.dams_df['AnimalID'].tolist())
            self.assertEqual(prog_num, 1) # Default progeny per mating

    def test_random_mating_n_matings_per_sire(self):
        n_mat_sire = 2
        matings = random_mating(self.sires_df, self.dams_df, sire_id_col='AnimalID', dam_id_col='AnimalID',
                                n_matings_per_sire=n_mat_sire, n_progeny_per_mating=1)

        # Check if each sire has at most n_mat_sire unique dams
        sire_dam_map = {}
        for sire, dam, _ in matings:
            if sire not in sire_dam_map: sire_dam_map[sire] = set()
            sire_dam_map[sire].add(dam)

        for sire in self.sires_df['AnimalID']:
            if sire in sire_dam_map: # Sire might not be used if not enough dams or max_total_matings hit early
                 self.assertLessEqual(len(sire_dam_map[sire]), n_mat_sire)
        # Total matings should be around N_sires * n_mat_sire if enough dams
        # Max possible here is 3 sires * 2 dams/sire = 6 matings (if enough unique dams for each)
        self.assertLessEqual(len(matings), len(self.sires_df) * n_mat_sire)


    def test_random_mating_max_total_matings(self):
        max_total = 3
        matings = random_mating(self.sires_df, self.dams_df, sire_id_col='AnimalID', dam_id_col='AnimalID',
                                max_total_matings=max_total, n_progeny_per_mating=1)
        # Number of unique pairs should be <= max_total
        unique_pairs = set((s,d) for s,d,_ in matings)
        self.assertLessEqual(len(unique_pairs), max_total)
        self.assertEqual(len(matings), len(unique_pairs)) # Since n_progeny_per_mating=1

    def test_random_mating_n_progeny_per_mating(self):
        n_prog = 3
        matings = random_mating(self.sires_df, self.dams_df, sire_id_col='AnimalID', dam_id_col='AnimalID',
                                max_total_matings=2, n_progeny_per_mating=n_prog)
        unique_pairs = set((s,d) for s,d,_ in matings)
        self.assertEqual(len(matings), len(unique_pairs) * n_prog)
        self.assertLessEqual(len(unique_pairs), 2)


    def test_random_mating_empty_inputs(self):
        empty_df = pd.DataFrame(columns=['AnimalID'])
        matings_no_sires = random_mating(empty_df, self.dams_df, sire_id_col='AnimalID', dam_id_col='AnimalID')
        self.assertEqual(len(matings_no_sires), 0)
        matings_no_dams = random_mating(self.sires_df, empty_df, sire_id_col='AnimalID', dam_id_col='AnimalID')
        self.assertEqual(len(matings_no_dams), 0)

    def test_random_mating_avoid_self_mating(self):
        # Test case where a sire could also be a dam
        overlapping_sires = pd.DataFrame({'ID': ['S1', 'S2', 'D1']}) # D1 is also a sire
        overlapping_dams = pd.DataFrame({'ID': ['D1', 'D2', 'S2']})  # S2 is also a dam

        # Run many times to increase chance of self-mating if not avoided
        found_self_mating = False
        for _ in range(50):
            matings = random_mating(overlapping_sires, overlapping_dams,
                                    sire_id_col='ID', dam_id_col='ID',
                                    avoid_self_mating=True, max_total_matings=5)
            for s, d, _ in matings:
                if s == d:
                    found_self_mating = True
                    break
            if found_self_mating: break
        self.assertFalse(found_self_mating)

        # Now allow self-mating and check if it can happen
        # This is probabilistic, so it's not a perfect test.
        # A more deterministic test would involve specific n_matings_per_sire with limited choices.
        # For now, this is a basic check.


    def test_calculate_progeny_inbreeding(self):
        # A(S1,D1) = 0.25 -> F_progeny = 0.5 * 0.25 = 0.125
        inbreeding1 = calculate_progeny_inbreeding('S1', 'D1', self.nrm_df)
        self.assertAlmostEqual(inbreeding1, 0.125)

        # A(S2,D2) = 0.0 -> F_progeny = 0.5 * 0.0 = 0.0
        inbreeding2 = calculate_progeny_inbreeding('S2', 'D2', self.nrm_df)
        self.assertAlmostEqual(inbreeding2, 0.0)

        # Sire not in NRM
        inbreeding_missing_sire = calculate_progeny_inbreeding('SX', 'D1', self.nrm_df)
        self.assertTrue(np.isnan(inbreeding_missing_sire))

        # Dam not in NRM
        inbreeding_missing_dam = calculate_progeny_inbreeding('S1', 'DX', self.nrm_df)
        self.assertTrue(np.isnan(inbreeding_missing_dam))


    def test_generate_mating_list_with_inbreeding(self):
        mating_pairs = [('S1', 'D1', 1), ('S2', 'D2', 1), ('S1', 'D2', 1)] # Sire, Dam, ProgNum

        inbreeding_df = generate_mating_list_with_inbreeding(mating_pairs, self.nrm_df)

        self.assertEqual(len(inbreeding_df), 3)
        self.assertIn('ProgenyInbreeding', inbreeding_df.columns)
        self.assertIn('Info1', inbreeding_df.columns) # For progeny number

        # Check specific values
        val_s1d1 = inbreeding_df[(inbreeding_df['SireID']=='S1') & (inbreeding_df['DamID']=='D1')]['ProgenyInbreeding'].iloc[0]
        self.assertAlmostEqual(val_s1d1, 0.125)

        val_s2d2 = inbreeding_df[(inbreeding_df['SireID']=='S2') & (inbreeding_df['DamID']=='D2')]['ProgenyInbreeding'].iloc[0]
        self.assertAlmostEqual(val_s2d2, 0.0)

        val_s1d2 = inbreeding_df[(inbreeding_df['SireID']=='S1') & (inbreeding_df['DamID']=='D2')]['ProgenyInbreeding'].iloc[0]
        # Assuming S1 and D2 are unrelated in self.nrm_df (default identity matrix part)
        self.assertAlmostEqual(val_s1d2, 0.0) # A(S1,D2) = 0 if not set otherwise

    def test_generate_mating_list_with_inbreeding_empty_list(self):
        inbreeding_df = generate_mating_list_with_inbreeding([], self.nrm_df)
        self.assertTrue(inbreeding_df.empty)
        self.assertIn('ProgenyInbreeding', inbreeding_df.columns)


if __name__ == '__main__':
    unittest.main()
