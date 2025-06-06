# sheep_breeding_genomics/data_management/data_structures.py

import pandas as pd

class PhenotypicData:
    """
    A structure to hold and manage phenotypic data.
    Currently, it's a wrapper around a Pandas DataFrame.
    """
    def __init__(self, data: pd.DataFrame = None):
        """
        Initializes the PhenotypicData object.

        Args:
            data (pd.DataFrame, optional): A DataFrame containing phenotypic data.
                                           Defaults to an empty DataFrame.
        """
        if data is None:
            self.data = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy() # Use a copy to avoid modifying the original DataFrame
        else:
            raise ValueError("PhenotypicData must be initialized with a Pandas DataFrame or None.")

    def __str__(self):
        return f"PhenotypicData with {self.data.shape[0]} records and {self.data.shape[1]} columns.\\n" + str(self.data.head())

    def get_summary(self) -> pd.DataFrame:
        """
        Returns a descriptive summary of the phenotypic data.
        """
        if not self.data.empty:
            return self.data.describe(include='all')
        else:
            return pd.DataFrame() # Return empty DataFrame if no data

    def add_record(self, record: dict):
        """
        Adds a new record to the phenotypic data.
        The record should be a dictionary where keys are column names.
        """
        if isinstance(record, dict):
            new_record_df = pd.DataFrame([record])
            self.data = pd.concat([self.data, new_record_df], ignore_index=True)
        else:
            raise ValueError("Record must be a dictionary.")

class PedigreeData:
    """
    A structure to hold and manage pedigree data.
    Currently, it's a wrapper around a Pandas DataFrame.
    Expected columns: AnimalID, SireID, DamID
    """
    def __init__(self, data: pd.DataFrame = None):
        """
        Initializes the PedigreeData object.

        Args:
            data (pd.DataFrame, optional): A DataFrame containing pedigree data.
                                           Defaults to an empty DataFrame.
                                           Expected columns: 'AnimalID', 'SireID', 'DamID'.
        """
        if data is None:
            self.data = pd.DataFrame(columns=['AnimalID', 'SireID', 'DamID'])
        elif isinstance(data, pd.DataFrame):
            # Basic check for essential pedigree columns
            required_cols = {'AnimalID', 'SireID', 'DamID'}
            if not required_cols.issubset(data.columns):
                raise ValueError(f"Pedigree DataFrame must contain columns: {required_cols}")
            self.data = data.copy() # Use a copy
        else:
            raise ValueError("PedigreeData must be initialized with a Pandas DataFrame or None.")

    def __str__(self):
        return f"PedigreeData with {self.data.shape[0]} records.\\n" + str(self.data.head())

    def get_summary(self) -> pd.DataFrame:
        """
        Returns a descriptive summary of the pedigree data.
        """
        if not self.data.empty:
            return self.data.describe(include='all')
        else:
            return pd.DataFrame()

    def add_individual(self, animal_id, sire_id=None, dam_id=None):
        """
        Adds a new individual to the pedigree.
        """
        new_individual = pd.DataFrame([{'AnimalID': animal_id, 'SireID': sire_id, 'DamID': dam_id}])
        self.data = pd.concat([self.data, new_individual], ignore_index=True)


if __name__ == '__main__':
    # Example Usage for PhenotypicData
    phen_dict = {'AnimalID': [101, 102, 103, 104],
                 'Weight': [50, 55, 52, 60],
                 'WoolYield': [5.1, 5.5, 5.0, 6.2]}
    phen_df = pd.DataFrame(phen_dict)

    phen_data_obj = PhenotypicData(phen_df)
    print("--- Phenotypic Data ---")
    print(phen_data_obj)
    print("\\nSummary:")
    print(phen_data_obj.get_summary())
    phen_data_obj.add_record({'AnimalID': 105, 'Weight': 58, 'WoolYield': 5.9})
    print("\\nAfter adding a record:")
    print(phen_data_obj)

    empty_phen_data = PhenotypicData()
    print("\\n--- Empty Phenotypic Data ---")
    print(empty_phen_data)
    print("\\nSummary (empty):")
    print(empty_phen_data.get_summary())

    # Example Usage for PedigreeData
    ped_dict = {'AnimalID': [101, 102, 103, 104, 201, 202],
                'SireID': [None, 101, 101, 102, None, 201],
                'DamID': [None, None, 102, None, None, 101]} # Using None for unknown parents
    ped_df = pd.DataFrame(ped_dict)

    ped_data_obj = PedigreeData(ped_df)
    print("\\n--- Pedigree Data ---")
    print(ped_data_obj)
    print("\\nSummary:")
    print(ped_data_obj.get_summary())
    ped_data_obj.add_individual(animal_id=203, sire_id=103, dam_id=202)
    print("\\nAfter adding an individual:")
    print(ped_data_obj)

    empty_ped_data = PedigreeData()
    print("\\n--- Empty Pedigree Data ---")
    print(empty_ped_data)
    empty_ped_data.add_individual(animal_id=301)
    print("\\nAfter adding an individual to empty pedigree:")
    print(empty_ped_data)
    print("\\nSummary (empty):")
    print(empty_ped_data.get_summary())

    # Example of initializing with incorrect DataFrame for PedigreeData
    try:
        bad_ped_df = pd.DataFrame({'ID': [1,2,3]})
        PedigreeData(bad_ped_df)
    except ValueError as e:
        print(f"\\nError caught as expected: {e}")


class GenomicData:
    """
    A structure to hold and manage SNP genotype data.
    Assumes genotypes are coded numerically (e.g., 0, 1, 2 for AA, AB, BB).
    Animals are rows, SNPs are columns. The first column is expected to be AnimalID.
    """
    def __init__(self, data: pd.DataFrame = None, animal_id_col: str = 'AnimalID'):
        """
        Initializes the GenomicData object.

        Args:
            data (pd.DataFrame, optional): DataFrame where rows are animals and columns are SNPs.
                                           The first column should be animal identifiers.
                                           Defaults to an empty DataFrame.
            animal_id_col (str): Name of the column containing Animal IDs.
        """
        if data is None:
            self.data = pd.DataFrame()
            self.animal_id_col = animal_id_col
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                self.data = pd.DataFrame() # Allow empty dataframe initialization
                self.animal_id_col = animal_id_col
            elif animal_id_col not in data.columns:
                raise ValueError(f"Animal ID column '{animal_id_col}' not found in the provided data.")
            else:
                self.data = data.copy()
                self.animal_id_col = animal_id_col
                # Consider setting AnimalID as index for easier operations, but keep it as a column for flexibility
                # self.data = data.set_index(animal_id_col, drop=False)
        else:
            raise ValueError("GenomicData must be initialized with a Pandas DataFrame or None.")

    @property
    def animal_ids(self) -> list:
        """Returns a list of animal IDs."""
        if not self.data.empty:
            return self.data[self.animal_id_col].tolist()
        return []

    @property
    def snp_names(self) -> list:
        """Returns a list of SNP names (column headers, excluding AnimalID)."""
        if not self.data.empty:
            return self.data.columns.drop(self.animal_id_col).tolist()
        return []

    @property
    def num_animals(self) -> int:
        """Returns the number of animals."""
        if not self.data.empty:
            return self.data.shape[0]
        return 0

    @property
    def num_snps(self) -> int:
        """Returns the number of SNPs."""
        if not self.data.empty:
            # Subtract 1 for the animal ID column
            return self.data.shape[1] - 1 if self.animal_id_col in self.data.columns else self.data.shape[1]
        return 0

    def get_genotypes(self) -> pd.DataFrame:
        """
        Returns the genotype matrix (excluding the animal ID column).
        """
        if not self.data.empty:
            return self.data.drop(columns=[self.animal_id_col])
        return pd.DataFrame()

    def get_summary(self) -> dict:
        """
        Returns a summary of the genomic data.
        """
        return {
            "num_animals": self.num_animals,
            "num_snps": self.num_snps,
            "animal_ids_preview": self.animal_ids[:5],
            "snp_names_preview": self.snp_names[:5]
        }

    def __str__(self):
        if self.data.empty:
            return "GenomicData is empty."
        return (f"GenomicData with {self.num_animals} animals and {self.num_snps} SNPs.\n"
                f"Animal ID column: '{self.animal_id_col}'\n"
                f"First 5 animals: {self.animal_ids[:5]}\n"
                f"First 5 SNPs: {self.snp_names[:5]}\n"
                f"DataFrame head:\n{self.data.head()}")


if __name__ == '__main__':
    # (Previous example usages for PhenotypicData and PedigreeData remain here)
    # ... (keep existing example printouts) ...
    print("\n--- Previous examples for Phenotypic and Pedigree Data above ---")

    # Example Usage for GenomicData
    print("\\n--- Genomic Data ---")
    geno_dict = {
        'AnimalID': ['Sheep001', 'Sheep002', 'Sheep003', 'Sheep004'],
        'SNP1': [0, 1, 2, 0],
        'SNP2': [1, 1, 0, 2],
        'SNP3': [2, np.nan, 1, 1], # Includes a missing value
        'SNP4': [0, 0, 1, 2]
    }
    geno_df = pd.DataFrame(geno_dict)

    genomic_data_obj = GenomicData(geno_df, animal_id_col='AnimalID')
    print(genomic_data_obj)
    print("\\nSummary:")
    print(genomic_data_obj.get_summary())
    print("\\nGenotypes only:")
    print(genomic_data_obj.get_genotypes().head())
    print(f"Animal IDs: {genomic_data_obj.animal_ids}")
    print(f"SNP Names: {genomic_data_obj.snp_names}")
    print(f"Num Animals: {genomic_data_obj.num_animals}")
    print(f"Num SNPs: {genomic_data_obj.num_snps}")


    empty_genomic_data = GenomicData(animal_id_col='SampleID')
    print("\\n--- Empty Genomic Data ---")
    print(empty_genomic_data)
    print("\\nSummary (empty):")
    print(empty_genomic_data.get_summary())

    # Example of initializing with error
    try:
        bad_geno_df = pd.DataFrame({'S1': [0,1], 'S2': [1,2]})
        GenomicData(bad_geno_df, animal_id_col='AnimalID') # AnimalID col missing
    except ValueError as e:
        print(f"\\nError caught as expected: {e}")

    # Example with different AnimalID column name
    geno_df_alt_id = pd.DataFrame({
        'SampleRef': ['S10', 'S11'],
        'MarkerA': [0,1],
        'MarkerB': [2,2]
    })
    genomic_data_obj_alt = GenomicData(geno_df_alt_id, animal_id_col='SampleRef')
    print("\\n--- Genomic Data with alternative ID column ---")
    print(genomic_data_obj_alt)
    print(genomic_data_obj_alt.get_summary())
