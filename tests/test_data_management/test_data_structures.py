# tests/test_data_management/test_data_structures.py

import unittest
import pandas as pd
import numpy as np

# Adjust path to import from the root of the project
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sheep_breeding_genomics.data_management.data_structures import PhenotypicData, PedigreeData, GenomicData

class TestPhenotypicData(unittest.TestCase):

    def test_phenotypic_data_initialization_empty(self):
        phen_data = PhenotypicData()
        self.assertTrue(phen_data.data.empty)

    def test_phenotypic_data_initialization_with_data(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'Trait1': [10.1, 12.3]})
        phen_data = PhenotypicData(data=df)
        pd.testing.assert_frame_equal(phen_data.data, df)
        self.assertNotEqual(id(phen_data.data), id(df)) # Ensure it's a copy

    def test_phenotypic_data_str_representation(self):
        df = pd.DataFrame({'AnimalID': [1], 'Trait1': [10.0]})
        phen_data = PhenotypicData(data=df)
        self.assertIn("PhenotypicData with 1 records and 2 columns.", str(phen_data))
        self.assertIn("AnimalID", str(phen_data))

    def test_phenotypic_data_get_summary(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'Trait1': [10.0, 20.0]})
        phen_data = PhenotypicData(data=df)
        summary = phen_data.get_summary()
        self.assertEqual(summary.loc['mean', 'Trait1'], 15.0)

    def test_phenotypic_data_add_record(self):
        phen_data = PhenotypicData()
        phen_data.add_record({'AnimalID': 1, 'Trait1': 10.5})
        self.assertEqual(phen_data.data.shape[0], 1)
        self.assertEqual(phen_data.data.loc[0, 'Trait1'], 10.5)
        with self.assertRaises(ValueError):
            phen_data.add_record("not a dict")


class TestPedigreeData(unittest.TestCase):

    def test_pedigree_data_initialization_empty(self):
        ped_data = PedigreeData()
        self.assertTrue(ped_data.data.empty)
        self.assertListEqual(list(ped_data.data.columns), ['AnimalID', 'SireID', 'DamID'])

    def test_pedigree_data_initialization_with_data(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'SireID': [0, 1], 'DamID': [0, 0]})
        ped_data = PedigreeData(data=df)
        pd.testing.assert_frame_equal(ped_data.data, df)

    def test_pedigree_data_initialization_missing_columns(self):
        df = pd.DataFrame({'AnimalID': [1, 2], 'SireID': [0, 1]}) # Missing DamID
        with self.assertRaises(ValueError):
            PedigreeData(data=df)

    def test_pedigree_data_str_representation(self):
        df = pd.DataFrame({'AnimalID': [1], 'SireID': [0], 'DamID': [0]})
        ped_data = PedigreeData(data=df)
        self.assertIn("PedigreeData with 1 records.", str(ped_data))

    def test_pedigree_data_get_summary(self):
        df = pd.DataFrame({'AnimalID': [1, 2, 3], 'SireID': [0, 1, 1], 'DamID': [0, 0, 2]})
        ped_data = PedigreeData(data=df)
        summary = ped_data.get_summary()
        self.assertEqual(summary.loc['count', 'AnimalID'], 3)

    def test_pedigree_data_add_individual(self):
        ped_data = PedigreeData()
        ped_data.add_individual(animal_id=1, sire_id=0, dam_id=0)
        self.assertEqual(ped_data.data.shape[0], 1)
        self.assertEqual(ped_data.data.loc[0, 'AnimalID'], 1)


class TestGenomicData(unittest.TestCase):

    def test_genomic_data_initialization_empty(self):
        gen_data = GenomicData(animal_id_col='ID')
        self.assertTrue(gen_data.data.empty)
        self.assertEqual(gen_data.animal_id_col, 'ID')

    def test_genomic_data_initialization_with_data(self):
        df = pd.DataFrame({'AnimalID': ['A1', 'A2'], 'SNP1': [0, 1], 'SNP2': [2, 0]})
        gen_data = GenomicData(data=df, animal_id_col='AnimalID')
        pd.testing.assert_frame_equal(gen_data.data, df)
        self.assertEqual(gen_data.num_animals, 2)
        self.assertEqual(gen_data.num_snps, 2)

    def test_genomic_data_initialization_missing_id_col(self):
        df = pd.DataFrame({'SNP1': [0, 1], 'SNP2': [2, 0]})
        with self.assertRaises(ValueError):
            GenomicData(data=df, animal_id_col='AnimalID')

    def test_genomic_data_properties(self):
        df = pd.DataFrame({'AID': ['A1', 'A2', 'A3'],
                           'snpA': [0,1,2], 'snpB': [1,1,0], 'snpC': [2,0,np.nan]})
        gen_data = GenomicData(data=df, animal_id_col='AID')
        self.assertListEqual(gen_data.animal_ids, ['A1', 'A2', 'A3'])
        self.assertListEqual(gen_data.snp_names, ['snpA', 'snpB', 'snpC'])
        self.assertEqual(gen_data.num_animals, 3)
        self.assertEqual(gen_data.num_snps, 3)

    def test_genomic_data_get_genotypes(self):
        df = pd.DataFrame({'AID': ['A1', 'A2'], 'snpA': [0,1], 'snpB': [1,1]})
        gen_data = GenomicData(data=df, animal_id_col='AID')
        genotypes_df = gen_data.get_genotypes()
        self.assertListEqual(list(genotypes_df.columns), ['snpA', 'snpB'])
        self.assertEqual(genotypes_df.shape, (2,2))

    def test_genomic_data_get_summary_dict(self):
        df = pd.DataFrame({'AnimalID': ['A1', 'A2'], 'SNP1': [0, 1]})
        gen_data = GenomicData(data=df, animal_id_col='AnimalID')
        summary = gen_data.get_summary()
        self.assertEqual(summary['num_animals'], 2)
        self.assertEqual(summary['num_snps'], 1)
        self.assertListEqual(summary['animal_ids_preview'], ['A1', 'A2'])

    def test_genomic_data_str_representation(self):
        df = pd.DataFrame({'AnimalID': ['A1'], 'SNP1': [0]})
        gen_data = GenomicData(data=df, animal_id_col='AnimalID')
        self.assertIn("GenomicData with 1 animals and 1 SNPs.", str(gen_data))

if __name__ == '__main__':
    unittest.main()
