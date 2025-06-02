=============
API Reference
=============

This section provides detailed documentation for the GenoStockPy Application Programming Interface (API).

Main Model Class
----------------

The primary interface for using GenoStockPy is the `GenoStockModel` class.

.. automodule:: genostockpy.api
   :members: GenoStockModel

Core Components
---------------

These are some of the core data structures and component classes that are managed internally
by `GenoStockModel` but are useful to understand.

.. automodule:: genostockpy.core.model_components
   :members: MME_py, VarianceComponent, GenotypesComponent, RandomEffectComponent

Pedigree Utilities
------------------

Functions and classes for handling pedigree information.

.. automodule:: genostockpy.pedigree.pedigree_module
   :members: Pedigree, PedNode, get_pedigree

Genotype Utilities
------------------

Functions for reading, processing, and handling genotype data.

.. automodule:: genostockpy.genotypes.genotype_handler
   :members: read_genotypes_py, calculate_grm_py

MCMC Engine
-----------

The core engine for running MCMC simulations. Typically used internally by `GenoStockModel.run()`.

.. automodule:: genostockpy.mcmc.mcmc_engine
   :members: run_mcmc_py

GWAS Module
-----------

Functionality for performing Genome-Wide Association Studies.

.. automodule:: genostockpy.gwas.gwas_module
   :members: run_window_gwas_py, calculate_marker_model_frequency_py

Dataset Loading
---------------

Utilities for loading example datasets.

.. automodule:: genostockpy.datasets.datasets_py
   :members: load_dataset_py

Output Utilities
----------------

Functions for saving and managing output from analyses.

.. automodule:: genostockpy.utils.output_utils
   :members: save_results_to_csv_py, get_ebv_py
