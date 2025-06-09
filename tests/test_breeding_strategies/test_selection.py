# tests/test_breeding_strategies/test_selection.py

import unittest
import pandas as pd
import numpy as np

# Adjust path to import from the root of the project
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.breeding_strategies.selection import (
    rank_animals,
    select_top_n,
    select_top_percent
)

class TestSelection(unittest.TestCase):

    def setUp(self):
        self.ebv_data = {
            'AnimalID': ['S1', 'D1', 'S2', 'D2', 'S3', 'D3'],
            'EBV': [10.5, 10.2, 11.0, 10.5, 9.8, 11.5], # S3 lowest, D3 highest, S1/D2 tie
            'OtherTrait': [1,2,3,4,5,6]
        }
        self.ebv_df = pd.DataFrame(self.ebv_data)
        self.ebv_df_indexed = self.ebv_df.set_index('AnimalID')

    def test_rank_animals_descending(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        self.assertEqual(ranked_df.iloc[0]['AnimalID'], 'D3') # Highest EBV
        self.assertEqual(ranked_df.iloc[-1]['AnimalID'], 'S3') # Lowest EBV
        self.assertIn('Rank', ranked_df.columns)
        self.assertEqual(ranked_df.iloc[0]['Rank'], 1)
        # Check tie handling - pandas sort is stable by default, order of tied elements maintained from original
        s1_rank = ranked_df[ranked_df['AnimalID'] == 'S1']['Rank'].iloc[0]
        d2_rank = ranked_df[ranked_df['AnimalID'] == 'D2']['Rank'].iloc[0]
        s2_rank = ranked_df[ranked_df['AnimalID'] == 'S2']['Rank'].iloc[0]
        # D3 (11.5) -> Rank 1
        # S2 (11.0) -> Rank 2
        # S1 (10.5) & D2 (10.5) -> Ranks 3 & 4 (order can vary based on original df)
        # D1 (10.2) -> Rank 5
        # S3 (9.8)  -> Rank 6
        self.assertEqual(ranked_df['Rank'].tolist(), [1,2,3,4,5,6])


    def test_rank_animals_ascending(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=True)
        self.assertEqual(ranked_df.iloc[0]['AnimalID'], 'S3') # Lowest EBV
        self.assertEqual(ranked_df.iloc[-1]['AnimalID'], 'D3') # Highest EBV
        self.assertEqual(ranked_df.iloc[0]['Rank'], 1)

    def test_rank_animals_with_index(self):
        ranked_df = rank_animals(self.ebv_df_indexed, ebv_col='EBV', ascending=False)
        self.assertEqual(ranked_df.index[0], 'D3')
        self.assertIn('Rank', ranked_df.columns)

    def test_rank_animals_empty_df(self):
        empty_df = pd.DataFrame(columns=['AnimalID', 'EBV'])
        ranked_df = rank_animals(empty_df, ebv_col='EBV', animal_id_col='AnimalID')
        self.assertTrue(ranked_df.empty)

    def test_rank_animals_missing_ebv_col(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='MissingEBV', animal_id_col='AnimalID')
        self.assertTrue(ranked_df.empty) # Expect empty on error

    def test_rank_animals_missing_animal_id_col_when_specified(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='MissingID')
        self.assertTrue(ranked_df.empty)


    def test_select_top_n(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)

        top_2 = select_top_n(ranked_df, 2)
        self.assertEqual(len(top_2), 2)
        self.assertEqual(top_2.iloc[0]['AnimalID'], 'D3')
        self.assertEqual(top_2.iloc[1]['AnimalID'], 'S2')

    def test_select_top_n_more_than_available(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_10 = select_top_n(ranked_df, 10)
        self.assertEqual(len(top_10), len(self.ebv_df)) # Select all available

    def test_select_top_n_zero(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_0 = select_top_n(ranked_df, 0)
        self.assertTrue(top_0.empty)
        self.assertListEqual(list(top_0.columns), list(ranked_df.columns))


    def test_select_top_n_negative(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_neg = select_top_n(ranked_df, -1)
        self.assertTrue(top_neg.empty) # Expect empty on error

    def test_select_top_n_empty_input_df(self):
        empty_ranked_df = pd.DataFrame(columns=['AnimalID', 'EBV', 'Rank'])
        selected = select_top_n(empty_ranked_df, 2)
        self.assertTrue(selected.empty)


    def test_select_top_percent(self):
        # 6 animals. Top 50% -> ceil(3) = 3 animals
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_50_percent = select_top_percent(ranked_df, 50.0)
        self.assertEqual(len(top_50_percent), 3)
        self.assertEqual(top_50_percent.iloc[0]['AnimalID'], 'D3')

    def test_select_top_percent_rounding(self):
        # Top 20% of 6 animals -> ceil(1.2) = 2 animals
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_20_percent = select_top_percent(ranked_df, 20.0)
        self.assertEqual(len(top_20_percent), 2)

    def test_select_top_percent_zero(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_0_percent = select_top_percent(ranked_df, 0.0)
        self.assertTrue(top_0_percent.empty)
        self.assertListEqual(list(top_0_percent.columns), list(ranked_df.columns))


    def test_select_top_percent_hundred(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        top_100_percent = select_top_percent(ranked_df, 100.0)
        self.assertEqual(len(top_100_percent), len(self.ebv_df))

    def test_select_top_percent_invalid_percentage(self):
        ranked_df = rank_animals(self.ebv_df, ebv_col='EBV', animal_id_col='AnimalID', ascending=False)
        selected_neg = select_top_percent(ranked_df, -10.0)
        self.assertTrue(selected_neg.empty)
        selected_over = select_top_percent(ranked_df, 110.0)
        self.assertTrue(selected_over.empty)

    def test_select_top_percent_empty_input_df(self):
        empty_ranked_df = pd.DataFrame(columns=['AnimalID', 'EBV', 'Rank'])
        selected = select_top_percent(empty_ranked_df, 10.0)
        self.assertTrue(selected.empty)


if __name__ == '__main__':
    unittest.main()
