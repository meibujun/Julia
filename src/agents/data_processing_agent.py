# src/agents/data_processing_agent.py

import pandas as pd
from src.core.datastructures import PhenotypeData, PedigreeData, GenotypeData

class DataProcessingAgent:
    def __init__(self):
        pass

    def load_phenotype_data(self, filepath: str) -> PhenotypeData:
        """
        Loads phenotype data from a CSV file.
        Expected columns: animal_id, trait_id, value
        """
        try:
            data = pd.read_csv(filepath)
            # Basic validation: Check for essential columns
            required_columns = ['animal_id', 'trait_id', 'value']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Phenotype CSV must contain columns: {', '.join(required_columns)}")
            print(f"Phenotype data loaded successfully from {filepath}")
            return PhenotypeData(data)
        except FileNotFoundError:
            print(f"Error: Phenotype file not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading phenotype data: {e}")
            raise

    def load_pedigree_data(self, filepath: str) -> PedigreeData:
        """
        Loads pedigree data from a CSV file.
        Expected columns: animal_id, sire_id, dam_id
        """
        try:
            data = pd.read_csv(filepath)
            # Basic validation: Check for essential columns
            required_columns = ['animal_id', 'sire_id', 'dam_id']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Pedigree CSV must contain columns: {', '.join(required_columns)}")
            print(f"Pedigree data loaded successfully from {filepath}")
            return PedigreeData(data)
        except FileNotFoundError:
            print(f"Error: Pedigree file not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading pedigree data: {e}")
            raise

    def load_genotype_data(self, filepath_data: str, filepath_marker_info: str = None) -> GenotypeData:
        """
        Loads genotype data from a CSV file.
        Rows are animals, columns are markers.
        Optionally loads marker information from another CSV.
        """
        try:
            # Assuming first column is animal_id
            genotypes = pd.read_csv(filepath_data)
            if 'animal_id' not in genotypes.columns:
                 # Fallback or error if animal_id is not a column name
                 # For now, let's assume the first column is implicitly animal_id if not named
                 # This part might need refinement based on common data formats
                 print("Warning: 'animal_id' column not found in genotype data. Assuming first column is animal ID.")
                 # A more robust solution would be to require an animal_id column or allow user to specify it.
                 # For this initial step, we'll proceed with caution.

            marker_info_df = None
            if filepath_marker_info:
                marker_info_df = pd.read_csv(filepath_marker_info)
                # Basic validation for marker_info if provided
                # e.g., required_marker_columns = ['marker_id', 'chromosome', 'position']
                # if not all(col in marker_info_df.columns for col in required_marker_columns):
                #     raise ValueError(f"Marker info CSV must contain columns: {', '.join(required_marker_columns)}")

            print(f"Genotype data loaded successfully from {filepath_data}")
            if filepath_marker_info:
                print(f"Marker info loaded successfully from {filepath_marker_info}")

            # The GenotypeData class expects the marker data itself (e.g., numpy array)
            # and marker_info (e.g., DataFrame).
            # We need to separate animal_id from the marker scores.
            if 'animal_id' in genotypes.columns:
                animal_ids = genotypes['animal_id']
                marker_data = genotypes.drop(columns=['animal_id'])
            else:
                # If no animal_id column, assume first column is ID and rest are markers
                # This is a simplification and might need to be made more robust
                animal_ids = genotypes.iloc[:, 0]
                marker_data = genotypes.iloc[:, 1:]


            # For GenotypeData, we might want to pass the animal_ids along with marker_data
            # or embed animal_ids as an index in a DataFrame if GenotypeData is modified to accept that.
            # For now, the GenotypeData constructor takes 'data' and 'marker_info'.
            # Let's adjust GenotypeData or how we pass data to it.
            # For simplicity here, we'll assume GenotypeData can handle a DataFrame with animal_id as index.
            # This implies a future modification to datastructures.py or a more complex setup here.
            # Let's assume marker_data is the numpy array of genotypes and marker_info_df is the marker info.
            # The animal_ids would need to be associated with the rows of marker_data.

            # This part needs careful alignment with how GenotypeData is actually implemented.
            # For now, we pass the raw genotypes DataFrame and marker_info DataFrame.
            return GenotypeData(genotypes, marker_info_df)

        except FileNotFoundError:
            print(f"Error: Genotype file not found at {filepath_data}")
            raise
        except Exception as e:
            print(f"Error loading genotype data: {e}")
            raise

# Example Usage (for testing purposes, would not be here in final agent)
if __name__ == '__main__':
    agent = DataProcessingAgent()

    # Create dummy CSV files for testing
    # Phenotype
    pheno_df = pd.DataFrame({
        'animal_id': [1, 2, 3, 1, 2],
        'trait_id': ['T1', 'T1', 'T1', 'T2', 'T2'],
        'value': [10.1, 11.5, 9.8, 100.3, 102.1]
    })
    pheno_df.to_csv('dummy_phenotypes.csv', index=False)

    # Pedigree
    ped_df = pd.DataFrame({
        'animal_id': [1, 2, 3],
        'sire_id': [0, 1, 1], # 0 for unknown parent
        'dam_id': [0, 0, 2]
    })
    ped_df.to_csv('dummy_pedigree.csv', index=False)

    # Genotypes
    geno_df = pd.DataFrame({
        'animal_id': [1,2,3],
        'marker1': [0,1,2],
        'marker2': [1,1,0],
        'marker3': [2,0,1]
    })
    geno_df.to_csv('dummy_genotypes.csv', index=False)

    marker_info_df = pd.DataFrame({
        'marker_id': ['marker1', 'marker2', 'marker3'],
        'chromosome': [1, 1, 2],
        'position': [100, 200, 50]
    })
    marker_info_df.to_csv('dummy_marker_info.csv', index=False)


    try:
        phenotypes = agent.load_phenotype_data('dummy_phenotypes.csv')
        print(f"Loaded {len(phenotypes.data)} phenotype records.")

        pedigree = agent.load_pedigree_data('dummy_pedigree.csv')
        print(f"Loaded {len(pedigree.data)} pedigree records.")

        genotypes = agent.load_genotype_data('dummy_genotypes.csv', 'dummy_marker_info.csv')
        print(f"Loaded genotype data for {len(genotypes.data)} animals and {len(genotypes.marker_info if genotypes.marker_info is not None else [])} markers.")

    except Exception as e:
        print(f"An error occurred during example usage: {e}")

    # Clean up dummy files
    import os
    os.remove('dummy_phenotypes.csv')
    os.remove('dummy_pedigree.csv')
    os.remove('dummy_genotypes.csv')
    os.remove('dummy_marker_info.csv')
