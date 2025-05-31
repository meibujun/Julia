===========================
Tutorial: Basic GBLUP Analysis
===========================

This tutorial guides you through performing a basic Genomic Best Linear Unbiased Prediction (GBLUP)
analysis using the `GenoStockModel` API.

Prerequisites
-------------

*   GenoStockPy installed (see :doc:`installation`).
*   Basic understanding of mixed models and genomic prediction.

Example Data
------------

For this tutorial, we will use a small example dataset. Assume this dataset is
available through the `load_dataset_py` utility, located in the `gblup_example`
dataset directory. It consists of:

*   `phenotypes.csv`: Contains individual IDs, trait measurements (e.g., 'Yield'),
    and potentially fixed effects (e.g., 'FixedEffect1').
*   `genotypes.csv`: Contains marker genotypes for individuals. First column is ID,
    followed by marker columns.
*   `pedigree.csv`: (Optional for pure GBLUP if all animals are genotyped, but can be
    included for context or if building A-matrix for comparison). Contains
    individual, sire, and dam IDs.

Step 1: Import necessary modules
--------------------------------

First, import the `GenoStockModel` from the API and the dataset loader.

.. code-block:: python

   from genostockpy.api import GenoStockModel
   from genostockpy.datasets import load_dataset_py # Assuming datasets_py.py is in genostockpy.datasets

Step 2: Load Data
-----------------

Use `load_dataset_py` to get the paths to your data files. Then, load them
using pandas or directly into `GenoStockModel` methods.

.. code-block:: python

   # Get paths to example data files
   # These paths would point to where your actual data files are.
   # For this example, we assume they are structured as described.
   try:
       pheno_file = load_dataset_py(dataset_name="gblup_example", file_name="phenotypes.csv")
       geno_file = load_dataset_py(dataset_name="gblup_example", file_name="genotypes.csv")
       # ped_file = load_dataset_py(dataset_name="gblup_example", file_name="pedigree.csv") # Optional
   except FileNotFoundError as e:
       print(f"Error loading example dataset files: {e}")
       print("Please ensure dummy data files exist in genostockpy/data/gblup_example/")
       # Exit or handle error appropriately in a real script
       raise

   # In a real scenario, you would load these files.
   # For this conceptual tutorial, we'll mostly show API usage.

Step 3: Initialize the Model
----------------------------

Create an instance of the `GenoStockModel`.

.. code-block:: python

   model = GenoStockModel(model_name="GBLUP_Tutorial_Analysis")

Step 4: Configure the Model
---------------------------

Define the model equation, load phenotypes, and add genotype information.

.. code-block:: python

   # 1. Define the model equation
   # Assuming 'Yield' is the trait and 'FixedEffect1' is a fixed effect.
   # 'animal' will represent the polygenic/genomic random effect.
   model.set_model_equation(
       equation="Yield = intercept + FixedEffect1 + animal",
       trait_types={"Yield": "continuous"}
   )

   # 2. Load phenotype data
   # In a real script, you'd pass the actual pheno_file path or a DataFrame.
   # pd.read_csv(pheno_file) would be used if pheno_file is a path.
   # For this example, assume pheno_df is a pre-loaded DataFrame if not using file paths.
   model.load_phenotypes(
       data=pheno_file, # Or a pandas DataFrame
       id_column="ID",
       # trait_columns=["Yield"], # Traits are now derived from set_model_equation
       covariate_columns=["FixedEffect1"], # FixedEffect1 is treated as a covariate here
       sep=',' # Assuming CSV with comma separator
   )

   # 3. (Optional) Load pedigree if you want to compare with PBLUP or for other reasons
   # model.load_pedigree(
   #     file_path=ped_file, # Or a pandas DataFrame
   #     header=True,      # Example: if pedigree file has a header
   #     separator=',',
   #     missing_strings=["0", "NA"]
   # )

   # 4. Add genotype data for GBLUP
   # For GBLUP, the 'data_source' provides the raw marker data to compute the GRM,
   # or it can be a pre-computed GRM if the method indicates it.
   # 'genetic_variance' is the prior for the total genetic variance explained by markers (sigma_a^2).
   model.add_genotypes(
       name="snp_chip",
       data_source=geno_file, # Or a pandas DataFrame or NumPy array
       method="GBLUP",
       # Priors for GBLUP (total genetic variance)
       genetic_variance=0.5, # Example prior value for sigma_a^2
       df_prior_g=5.0,         # Degrees of freedom for the prior
       # Options for reading the genotype file (if data_source is a path)
       header=True, separator=',', missing_value_code="9",
       # QC and processing options
       perform_qc=True, maf_threshold=0.01,
       center_genotypes=True # Centering is part of GRM calculation (M-P)
   )

   # 5. Define the 'animal' random effect (polygenic/genomic based on GBLUP)
   # For GBLUP, the genetic effect is tied to the GenotypesComponent.
   # If 'animal' in the equation refers to the GBLUP effects, its variance comes
   # from the 'genetic_variance' set in add_genotypes.
   # If a separate pedigree-based polygenic effect was also desired (e.g. in a model like y = fixed + animal_ped + animal_geno),
   # you would add it explicitly here with use_pedigree=True.
   # For a simple GBLUP (y = fixed + g), where g ~ N(0, G*sigma_g^2), the 'animal' term
   # in the equation represents 'g'. The link is made via the method in add_genotypes.
   # No separate add_random_effect for 'animal' is needed if it's purely GBLUP from one GenotypesComponent.
   # However, if 'animal' was to be a pedigree effect AND you also had SNP effects, you'd define 'animal' here
   # and marker effects via add_genotypes(method="BayesC" etc.)
   # For GBLUP/SSGBLUP, 'animal' typically becomes the term whose variance is sigma_g^2 based on G or H.
   # The current API implies that `add_genotypes` with method "GBLUP" handles the setup for the genomic random effect.
   # The term used in `set_model_equation` (e.g., "animal") needs to be understood by the backend
   # as referring to this genomic effect. This linkage is conceptual in `_prepare_for_run`.

   # 6. Set MCMC options
   model.set_mcmc_options(
       chain_length=5000,
       burn_in=1000,
       thinning=5,
       seed=123 # For reproducibility
   )

Step 5: Run the Analysis
------------------------

Execute the MCMC analysis.

.. code-block:: python

   # This step will internally:
   # 1. Call _prepare_for_run() to:
   #    - Parse the model equation.
   #    - Process phenotype data.
   #    - Process pedigree data (if loaded).
   #    - Process genotype data:
   #        - Read markers (from geno_file).
   #        - Perform QC (MAF filter, imputation).
   #        - Center markers.
   #        - Calculate GRM (for GBLUP).
   #    - Configure GenotypesComponent for GBLUP with the GRM and priors.
   #    - Set up other random/fixed effects.
   #    - Validate data and set default priors.
   #    - Build initial MME matrices.
   #    - Initialize MCMC state.
   # 2. Call the MCMC engine (run_mcmc_py).

   print("Running MCMC analysis (conceptual)...")
   # results = model.run(output_folder="gblup_tutorial_results")
   # print("Analysis finished. Results are stored.")

   # Note: Since the MCMC engine's statistical sampling is not fully implemented,
   # running this will primarily test the API flow and data setup orchestration.

Step 6: Retrieve and Interpret Results
--------------------------------------

After the analysis, you can access results like EBVs and variance components.

.. code-block:: python

   print("Retrieving results (conceptual)...")
   # ebvs = model.get_ebv()
   # if ebvs is not None:
   #     print("\nEstimated Breeding Values (EBVs):")
   #     print(ebvs.head())

   # variance_components = model.get_variance_components()
   # if variance_components is not None:
   #     print("\nVariance Components:")
   #     for component, value in variance_components.items():
   #         print(f"  {component}: {value}")

   # model.summary()

   # Example of expected output structure (conceptual)
   # EBVs: DataFrame with ID, EBV_Yield, PEV_Yield
   # Variance Components: Dict like {'ResidualVariance': ..., 'snp_chip_genetic_variance': ...}

This tutorial demonstrates the basic workflow for setting up and (conceptually)
running a GBLUP analysis with GenoStockPy. As the library develops, the internal
MCMC computations will be fully implemented, and result interpretation will become
more concrete.
```

Now, creating `api_reference.rst`. This will use `automodule` to pull documentation from `genostockpy.api`.
