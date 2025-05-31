============
Installation
============

This guide provides instructions on how to install GenoStockPy and its dependencies.

Prerequisites
-------------

*   **Python**: GenoStockPy requires Python 3.8 or later.
*   **pip**: Python's package installer, usually included with Python.

Dependencies
------------

GenoStockPy relies on the following Python libraries:

*   **NumPy**: For numerical operations, especially array manipulation.
*   **Pandas**: For data manipulation and analysis, particularly using DataFrames.
*   **SciPy**: For scientific and technical computing, including sparse matrices and linear algebra.

These dependencies will be automatically installed if you install GenoStockPy using pip (once a `pyproject.toml` or `setup.py` with these dependencies listed is available).

Installation Methods
--------------------

As GenoStockPy is currently under development, standard installation from PyPI might not be available yet.

Method 1: Installing from a Git Repository (Conceptual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the project is hosted on a Git repository (e.g., GitHub), you can install it directly:

.. code-block:: bash

   pip install git+https://github.com/yourusername/genostockpy.git#egg=genostockpy

Replace `yourusername/genostockpy.git` with the actual repository URL.

Method 2: Installing from a Local Clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a local copy of the GenoStockPy source code:

1.  **Clone the repository (if you haven't already):**

    .. code-block:: bash

       git clone https://github.com/yourusername/genostockpy.git
       cd genostockpy

2.  **Install using pip:**

    From the root directory of the cloned project (where `pyproject.toml` or `setup.py` is located), run:

    .. code-block:: bash

       pip install .

    For development mode, which allows you to make changes to the source code that are immediately reflected without reinstalling:

    .. code-block:: bash

        pip install -e .

Verifying Installation
----------------------

After installation, you can verify it by opening a Python interpreter and trying to import the package:

.. code-block:: python

   import genostockpy
   # print(genostockpy.__version__) # Assuming a __version__ attribute is set in genostockpy/__init__.py

If the import succeeds, the installation was successful.

Troubleshooting
---------------

*   **Missing Dependencies**: If pip fails to install dependencies, you might need to install them manually first:
    .. code-block:: bash

       pip install numpy pandas scipy

*   **Permissions**: If you encounter permission errors, try using `pip install --user .` or ensure you have the necessary write permissions for the Python environment.
