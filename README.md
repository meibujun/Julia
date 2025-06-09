# Sheep Breeding Genomics Toolkit

## Overview

The Sheep Breeding Genomics Toolkit is a Python-based suite of tools designed to simulate and analyze data relevant to sheep breeding programs. It provides functionalities for managing phenotypic, pedigree, and genomic data, performing genetic evaluations (PBLUP, GBLUP, ssGBLUP), implementing selection strategies, and simulating mating schemes including inbreeding calculation. The toolkit is structured into several modules to facilitate a clear workflow from data input to breeding program simulation.

## Directory Structure

The project is organized into the following key directories:

-   `sheep_breeding_genomics/`: Contains the core library code.
    -   `data_management/`: Modules for creating, handling, and validating phenotypic, pedigree, and genomic data structures.
    -   `genetic_evaluation/`: Modules for calculating relationship matrices (NRM, GRM, H_inv) and solving Mixed Model Equations for BLUP, GBLUP, and ssGBLUP.
    -   `breeding_strategies/`: Modules for animal selection strategies and mating scheme simulations.
-   `examples/`: Contains example scripts demonstrating how to use the toolkit, including a comprehensive breeding program simulation.
-   `README.md`: This file.
-   `requirements.txt`: Lists project dependencies.

## Modules Overview

### 1. Data Management (`sheep_breeding_genomics.data_management`)

This module is responsible for handling the various types of data used in animal breeding:

-   **`data_structures.py`**: Defines classes (`PhenotypicData`, `PedigreeData`, `GenomicData`) to hold and manage dataframes for phenotypic records, pedigree information, and SNP genotypes.
-   **`io_handlers.py`**: Provides functions to read data from files (e.g., CSV) into the defined data structures and to save them.
-   **`validators.py`**: Offers functions for validating the integrity and quality of the data (e.g., checking for missing values, valid genotype codes, call rates, MAF).

### 2. Genetic Evaluation (`sheep_breeding_genomics.genetic_evaluation`)

This module implements core genetic evaluation methods:

-   **`relationship_matrix.py`**:
    -   `calculate_nrm()`: Calculates the Numerator Relationship Matrix (A) from pedigree data.
    -   `calculate_grm()`: Calculates the Genomic Relationship Matrix (G) from SNP data (VanRaden's Method 1).
    -   `calculate_h_inverse_matrix()`: Calculates the inverse of the H matrix (H<sup>-1</sup>) required for Single-Step GBLUP (ssGBLUP), combining pedigree and genomic information.
-   **`blup_models.py`**:
    -   `solve_animal_model_mme()`: Solves Mixed Model Equations (MME) for animal models using either NRM (for PBLUP) or GRM (for GBLUP) to estimate (Genomic) Estimated Breeding Values ((G)EBVs).
    -   `solve_ssgblup_model_mme()`: Solves MME for Single-Step GBLUP using a pre-computed H<sup>-1</sup> matrix to estimate ssGEBVs.

### 3. Breeding Strategies (`sheep_breeding_genomics.breeding_strategies`)

This module provides tools to simulate selection and mating decisions:

-   **`selection.py`**:
    -   `rank_animals()`: Ranks animals based on their (G)EBVs.
    -   `select_top_n()`: Selects a specified number of top animals.
    -   `select_top_percent()`: Selects a specified percentage of top animals.
-   **`mating_schemes.py`**:
    -   `random_mating()`: Generates random mating pairs from selected sires and dams.
    -   `calculate_progeny_inbreeding()`: Calculates the expected inbreeding coefficient of progeny from a sire and dam using their relationship from an NRM.
    -   `generate_mating_list_with_inbreeding()`: Appends expected progeny inbreeding to a list of mating pairs.

## Dependencies

The toolkit relies on the following Python libraries:

-   `pandas`
-   `numpy`
-   `scipy`

These are listed in the `requirements.txt` file.

## Getting Started / Examples

Each Python module file (`.py`) within the `sheep_breeding_genomics` subdirectories contains an `if __name__ == '__main__':` block with example usage for the functions and classes defined in that file. These provide direct illustrations of how to use individual components.

### How to Run Module Examples

To run these self-contained examples, navigate to the project's root directory (`sheep_breeding_genomics`) and use the `python -m` option. For example:

```bash
# From the project root directory
python -m sheep_breeding_genomics.data_management.io_handlers
python -m sheep_breeding_genomics.genetic_evaluation.relationship_matrix
python -m sheep_breeding_genomics.breeding_strategies.selection
# etc.
```

### Comprehensive Simulation Example

A more comprehensive example demonstrating a multi-generational breeding program simulation can be found in:

`examples/basic_breeding_program_simulation.py`

This script shows how to integrate components from all modules to:
1.  Initialize a base population (pedigree, phenotypes, true BVs).
2.  Iteratively perform genetic evaluation (PBLUP in the example).
3.  Select top sires and dams.
4.  Generate mating lists and calculate expected progeny inbreeding.
5.  Simulate progeny with new phenotypes and updated pedigree.
6.  Track genetic gain and inbreeding over generations.

To run this simulation, navigate to the project root directory and execute:

```bash
python examples/basic_breeding_program_simulation.py
```

## Limitations and Future Work (Optional)

-   **Variance Component Estimation:** The current toolkit assumes variance components are known. Future work could include REML estimation.
-   **Advanced Mating Schemes:** Only random mating is implemented. Other schemes like optimal contribution selection or mating to minimize inbreeding could be added.
-   **Genotype Simulation:** The basic simulation script does not simulate genotypes for new progeny; this would be needed for iterative GBLUP/ssGBLUP over generations.
-   **Performance:** For very large datasets, direct matrix inversions (A<sup>-1</sup>, G<sup>-1</sup>) can be computationally intensive. Sparse matrix operations or iterative solvers (e.g., for MME) could be explored for optimization.
-   **Error Handling:** While basic checks are in place, error handling could be made more robust.
-   **Input File Formats:** Currently primarily uses CSVs. Support for other common genetic data formats could be added.

## Testing

The project includes a suite of unit tests to verify the functionality of the core toolkit modules.

**Running Tests:**

To run all unit tests, navigate to the project root directory and execute the following command:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

This command will automatically discover and run all test files (matching the pattern `test_*.py`) within the `tests` directory and its subdirectories.

Make sure you have any necessary testing libraries installed (e.g., `unittest` is part of the standard library, but if `pytest` were used, it would be a dependency).
The tests cover modules for data management, genetic evaluation (relationship matrices, BLUP models), and breeding strategies.

For more details on the API testing and maintainer-specific information, refer to the `README.md` file located in the `sheep_breeding_api/` directory.
```
