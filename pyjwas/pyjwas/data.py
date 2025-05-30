import numpy as np
import pandas as pd

"""
Manages loading, validation, and access for various types of data 
used in genomic analyses, including phenotype, pedigree, and genotype data.
"""
import numpy as np
import pandas as pd

class PhenotypeData:
    """
    Handles phenotype data, including loading from CSV files and providing
    access to observations and design matrices.
    """
    def __init__(self, file_path: str):
        """
        Initializes PhenotypeData with the path to a phenotype data file.

        Args:
            file_path (str): Path to the CSV file containing phenotype data.
                             The file is expected to have a header row. An 'ID'
                             column is recommended for individual identification.
        """
        self.file_path = file_path
        self.data = None # Should be a pandas DataFrame after loading
        self.individual_ids = None
        self.load_data(file_path)

    def load_data(self, file_path: str):
        """
        Loads phenotype data from the specified CSV file into a pandas DataFrame.
        Populates `self.individual_ids` if an 'ID' column is present.

        Args:
            file_path (str): Path to the CSV file.
        
        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file is empty or there's an error during loading.
        """
        try:
            self.data = pd.read_csv(file_path)
            if 'ID' in self.data.columns:
                self.individual_ids = self.data['ID'].tolist()
            else:
                print("Warning: No 'ID' column found in phenotype data. Individual IDs will not be automatically populated.")
            print(f"PhenotypeData: Successfully loaded {file_path}. Shape: {self.data.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Phenotype data file not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Phenotype data file is empty: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading phenotype data from {file_path}: {e}")


    def validate_data(self):
        if self.data is None:
            print("Phenotype data not loaded.")
            return False
        # Add more specific validation as needed
        print(f"PhenotypeData: validate_data called. Data loaded with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        return True

    def get_y_vector(self, y_col_name):
        """
        Returns the phenotype vector as a NumPy array for the specified column.

        Args:
            y_col_name (str): The name of the column in the phenotype data DataFrame
                              that contains the dependent variable (phenotype values).

        Returns:
            np.ndarray: A NumPy array containing the phenotype values.

        Raises:
            ValueError: If data is not loaded or `y_col_name` is not found,
                        or if the column contains non-numeric data that cannot be coerced.
        """
        if self.data is None:
            raise ValueError("Phenotype data not loaded. Call load_data() first.")
        if y_col_name not in self.data.columns:
            raise ValueError(f"Column '{y_col_name}' not found in phenotype data. Available columns: {self.data.columns.tolist()}")
        
        y_vector = self.data[y_col_name].to_numpy()
        # Ensure y_vector is numeric and handle potential NaNs
        if not np.issubdtype(y_vector.dtype, np.number):
            try:
                y_vector = pd.to_numeric(y_vector, errors='raise')
            except ValueError as e:
                raise ValueError(f"Phenotype column '{y_col_name}' contains non-numeric data that cannot be coerced: {e}")
        
        if np.isnan(y_vector).any():
            # Option 1: Raise error
            # raise ValueError(f"Phenotype column '{y_col_name}' contains NaN values. Please clean the data.")
            # Option 2: Remove NaNs and corresponding individuals (would require alignment with X and Z)
            # For now, let downstream processes handle NaNs if they can, or raise error.
            print(f"Warning: Phenotype column '{y_col_name}' contains NaN values.")

        return y_vector

    def get_design_matrix_X(self, fixed_effect_terms):
        """
        Constructs and returns the design matrix X for specified fixed effect terms.
        
        The matrix X typically includes a column of ones for the intercept
        and columns for each fixed effect specified. Data for fixed effects
        is retrieved from the loaded phenotype data.

        Args:
            fixed_effect_terms (list[str]): A list of strings representing the fixed
                                            effect terms. "intercept" (case-insensitive)
                                            will add an intercept column. Other terms
                                            should match column names in the loaded data.

        Returns:
            np.ndarray: The design matrix X.

        Raises:
            ValueError: If data is not loaded or if a term is not found/valid.
                        Also if a data column for a term is non-numeric.
        """
        if self.data is None:
            raise ValueError("Phenotype data not loaded. Call load_data() first.")
        
        X_list = []
        processed_terms = []

        for term in fixed_effect_terms:
            if term.lower() == "intercept":
                if "intercept" not in processed_terms:
                    X_list.append(np.ones((self.data.shape[0], 1)))
                    processed_terms.append("intercept")
            elif term in self.data.columns:
                if term not in processed_terms:
                    term_data = self.data[term].to_numpy().reshape(-1, 1)
                    if not np.issubdtype(term_data.dtype, np.number):
                        try:
                            term_data = pd.to_numeric(term_data.flatten(), errors='raise').reshape(-1,1)
                        except ValueError as e:
                            raise ValueError(f"Fixed effect term '{term}' column contains non-numeric data: {e}")
                    X_list.append(term_data)
                    processed_terms.append(term)
            else:
                # This could be an interaction term or an error
                # For now, assume only direct column terms or "intercept"
                raise ValueError(f"Fixed effect term '{term}' not found in data columns or as 'intercept'. Available: {self.data.columns.tolist()}")

        if not X_list:
            # Should not happen if "intercept" is usually present or terms are validated by ModelDefinition
            return np.empty((self.data.shape[0], 0)) 
            
        X = np.hstack(X_list)
        # Handle NaNs in X, e.g., by mean imputation or raising an error
        if np.isnan(X).any():
            # For simplicity, impute with mean. More robust solutions might be needed.
            print("Warning: NaNs found in design matrix X. Imputing with column means.")
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
            if np.isnan(X).any(): # If a whole column was NaN
                print("Warning: Columns with all NaNs found in X after mean imputation. Filling with 0.")
                X = np.nan_to_num(X, nan=0.0)


        return X

    def get_design_matrix_Z(self, random_effect_terms, all_genotyped_ids_ordered):
        """
        Constructs and returns the design matrix Z for random effects.

        For GBLUP, this matrix maps genomic breeding values (which are typically
        estimated for all individuals present in the GRM) to the phenotyped
        individuals. If phenotype individual IDs are available and match those in
        `all_genotyped_ids_ordered`, a mapping is performed. Otherwise, simpler
        assumptions about direct correspondence are made (with warnings).

        Args:
            random_effect_terms (list[str]): A list of random effect terms. For simple
                                             GBLUP, this is often a placeholder like ["g"]
                                             as the structure is largely defined by the
                                             correspondence between phenotype and genotype IDs.
            all_genotyped_ids_ordered (list or np.array): An ordered list/array of
                                                          individual IDs that correspond to
                                                          the rows and columns of the GRM.

        Returns:
            np.ndarray: The design matrix Z.

        Raises:
            ValueError: If phenotype data is not loaded.
            NotImplementedError: If Z matrix construction is attempted for complex random
                                 effects without phenotype IDs.
        """
        if self.data is None:
            raise ValueError("Phenotype data not loaded.")
        if self.individual_ids is None:
            # If phenotype IDs are not available, we assume a direct match in order and count.
            # This is a strong assumption and often not true in practice.
            print("Warning: Phenotype individual IDs not available. Assuming direct 1:1 mapping and order " +
                  "between phenotypes and the first N rows of GRM, where N is number of phenotypes.")
            if len(random_effect_terms) > 1 or random_effect_terms[0] != "g": # Assuming 'g' is the GBLUP term
                 raise NotImplementedError("Z matrix construction for complex random effects without phenotype IDs is not supported.")
            
            n_pheno = self.data.shape[0]
            n_geno = len(all_genotyped_ids_ordered)
            if n_pheno > n_geno:
                raise ValueError("More phenotypes than genotypes, but no phenotype IDs to map. Cannot construct Z.")
            
            # Assume the first n_pheno individuals in GRM correspond to the phenotyped individuals
            Z = np.eye(n_pheno, n_geno) # Selects the first n_pheno genotype effects for the n_pheno phenotypes
            return Z

        # If phenotype IDs are available, perform mapping
        n_pheno = len(self.individual_ids)
        n_geno = len(all_genotyped_ids_ordered)
        
        Z = np.zeros((n_pheno, n_geno))
        
        # Create a mapping from genotype ID to its index in the GRM
        geno_id_to_index = {gid: i for i, gid in enumerate(all_genotyped_ids_ordered)}
        
        unmapped_pheno_ids = []
        for i, pheno_id in enumerate(self.individual_ids):
            if pheno_id in geno_id_to_index:
                geno_idx = geno_id_to_index[pheno_id]
                Z[i, geno_idx] = 1
            else:
                unmapped_pheno_ids.append(pheno_id)
        
        if unmapped_pheno_ids:
            print(f"Warning: {len(unmapped_pheno_ids)} phenotyped individuals were not found in the genotype ID list. " +
                  "Their rows in Z will be all zeros. Unmapped IDs: " + ", ".join(map(str,unmapped_pheno_ids[:5])) + ("..." if len(unmapped_pheno_ids)>5 else ""))
            # Depending on the model, this might be an error or handled by downstream processes (e.g. if these individuals have no records for this random effect)

        # This current Z construction is for a single random effect 'g' that applies to all individuals.
        # If random_effect_terms specified multiple random effects or complex structures, this would need refinement.
        if len(random_effect_terms) != 1 : # or random_effect_terms[0] not conventionally "g" or "animal"
             print(f"Warning: Z matrix construction assumes a single genomic random effect. Effect terms provided: {random_effect_terms}")

        return Z


class PedigreeData:
    """
    Handles pedigree data. (Currently a placeholder implementation).
    
    Future implementations would load pedigree information (e.g., individual, sire, dam)
    and could construct relationship matrices (e.g., A matrix) or perform pedigree validation.
    """
    def __init__(self, file_path: str):
        """
        Initializes PedigreeData with the path to a pedigree data file.

        Args:
            file_path (str): Path to the CSV file containing pedigree data.
        """
        self.file_path = file_path
        self.data = None # Initialize data attribute
        # Add self.individual_ids for PedigreeData if it's relevant for its use cases
        # self.individual_ids = None 
        self.load_data(file_path)

    def load_data(self, file_path: str):
        """
        Loads pedigree data from the specified CSV file.
        (Currently a placeholder - does not actually load or parse pedigree structure).

        Args:
            file_path (str): Path to the CSV file.
        """
        # Placeholder for PedigreeData loading
        # Example:
        # try:
        #     self.data = pd.read_csv(file_path) # Assuming CSV like others
        #     if 'ID' in self.data.columns: # Or relevant ID columns for pedigree
        #         self.individual_ids = self.data['ID'].tolist() 
        # except Exception as e:
        #     raise ValueError(f"Error loading pedigree data from {file_path}: {e}")
        print(f"PedigreeData: load_data called with {file_path}. Placeholder - actual data loading not implemented.")
        self.data = f"Data from {file_path}" # Placeholder assignment


    def validate_data(self):
        # Placeholder
        print("PedigreeData: validate_data called (placeholder).")
        pass

class GenotypeData:
    """
    Handles genotype data, including loading from CSV files, basic imputation,
    and calculation of the Genomic Relationship Matrix (GRM).
    """
    def __init__(self, file_path: str):
        """
        Initializes GenotypeData with the path to a genotype data file.

        Args:
            file_path (str): Path to the CSV file containing genotype data.
                             The first column is expected to be 'ID' (individual IDs),
                             and subsequent columns should be marker data (e.g., coded as 0, 1, 2).
        """
        self.file_path = file_path
        self.individual_ids = None # List of individual IDs, should match GRM order
        self.genotype_matrix = None # Numpy array (individuals x markers)
        self.grm = None # Numpy array (GRM)
        self.load_data(file_path)

    def load_data(self, file_path: str):
        """
        Loads genotype data from a CSV file.
        
        The method assumes the first column of the CSV is 'ID' (individual identifiers)
        and the remaining columns are marker data (e.g., coded numerically as 0, 1, 2).
        It populates `self.individual_ids` and `self.genotype_matrix`.
        Basic imputation for missing marker data is performed (mean imputation, then zero-fill).

        Args:
            file_path (str): Path to the CSV file.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file is empty, data cannot be converted to numeric,
                        or other loading errors occur.
        """
        try:
            data_df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Genotype data file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading genotype data from {file_path}: {e}")

        if data_df.empty:
            raise ValueError(f"Genotype data file {file_path} is empty.")

        # Assume first column is ID
        self.individual_ids = data_df.iloc[:, 0].values.tolist()
        
        # Attempt to convert remaining columns to numeric, coercing errors to NaN
        genotype_df = data_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        
        if genotype_df.isnull().any().any():
            # Simple strategy: fill NaNs with mean of the column (marker)
            # More sophisticated imputation might be needed for real data
            print(f"Warning: Missing values found in genotype data. Imputing with column means.")
            genotype_df = genotype_df.fillna(genotype_df.mean())
            # If after mean imputation, NaNs still exist (e.g., a whole column was NaN), raise error or fill with global mean/zero
            if genotype_df.isnull().any().any():
                 print(f"Warning: Columns with all NaNs found after mean imputation. Filling with 0.")
                 genotype_df = genotype_df.fillna(0)


        self.genotype_matrix = genotype_df.to_numpy()

        if not np.issubdtype(self.genotype_matrix.dtype, np.number):
            raise ValueError("Genotype matrix must be numeric after loading and coercion.")
        
        print(f"GenotypeData: Loaded {self.genotype_matrix.shape[0]} individuals and {self.genotype_matrix.shape[1]} markers.")


    def validate_data(self):
        """
        Validates the loaded genotype data.
        """
        if self.genotype_matrix is None:
            print("Genotype data not loaded. Call load_data() first.")
            return False
        
        # Example validation: check for non-numeric values if not handled during load
        if not np.issubdtype(self.genotype_matrix.dtype, np.number):
            print("Error: Genotype matrix contains non-numeric values.")
            return False
        
        # Example: Check for consistent number of markers per individual (already handled by DataFrame structure)
        # Example: Check for extreme allele frequencies if applicable
        print("GenotypeData: validate_data called. Basic checks passed.")
        return True

    def calculate_grm(self, method="vanraden"):
        """
        Calculates the Genomic Relationship Matrix (GRM) using specified methods.

        Currently, only the "vanraden" method is implemented.
        The calculated GRM is stored in `self.grm` and also returned.

        Args:
            method (str, optional): The method for GRM calculation.
                                    Defaults to "vanraden".

        Returns:
            np.ndarray: The calculated Genomic Relationship Matrix.

        Raises:
            ValueError: If the genotype matrix is not loaded or contains no markers.
            NotImplementedError: If a method other than "vanraden" is specified.
        """
        if self.genotype_matrix is None:
            raise ValueError("Genotype matrix not loaded. Call load_data() first.")

        if method == "vanraden":
            M = self.genotype_matrix # Shape: (n_individuals, n_markers)
            n_individuals, n_markers = M.shape

            if n_markers == 0:
                raise ValueError("Genotype matrix has no markers (columns).")

            # Calculate allele frequencies p_j for each marker j
            # Sum of allele counts / (2 * number of individuals)
            # Assuming M contains counts like 0, 1, 2 for diploid organisms
            p_j = np.sum(M, axis=0) / (2 * n_individuals) 
            
            # Handle cases where p_j is 0 or 1 to avoid issues in denominator (though sum should be okay)
            # p_j = np.clip(p_j, 1e-6, 1 - 1e-6) # Optional: if individual p_j*(1-p_j) is used

            # Create matrix P where each element P_ij = 2 * p_j
            # P should have the same dimensions as M. P_ij is 2*p_j for marker j, repeated for all individuals i.
            P = np.full_like(M, fill_value=2 * p_j, dtype=np.float64)
            
            Z = M - P
            
            # Calculate the denominator: 2 * sum(p_j * (1 - p_j))
            denom = 2 * np.sum(p_j * (1 - p_j))
            
            if denom == 0:
                # This happens if all markers are monomorphic (all p_j are 0 or 1)
                # Or if there's only one marker and it's monomorphic.
                # Consider returning an identity matrix or matrix of zeros, or raising a more specific error.
                print("Warning: Denominator for GRM calculation is zero (all markers may be monomorphic). Returning an identity matrix.")
                self.grm = np.eye(n_individuals) 
                return self.grm

            # Calculate GRM: G = (Z @ Z.T) / denom
            self.grm = (Z @ Z.T) / denom
            print(f"GRM calculated using VanRaden method. Shape: {self.grm.shape}")
            return self.grm
        else:
            raise NotImplementedError(f"GRM calculation method '{method}' is not implemented.")
