# Python Interface for GenomicPrediction.jl

This directory will contain the Python interface for the `GenomicPrediction.jl` package.

## Plan

The interface will be provided as a Python package that uses `PyJulia` to interact with the Julia backend.

### Key Features:
-   A scikit-learn compatible estimator class for each model type (e.g., `GBLUP`, `BayesA`).
-   These estimators will implement the standard `.fit(X, y)` and `.predict(X)` methods.
-   The `.fit()` method will convert pandas DataFrames or NumPy arrays into Julia objects, instantiate the corresponding Julia model, and call the `fit!` method.
-   The `.predict()` method will call the Julia `predict` function and return the results as a NumPy array.
-   Utility functions for easy data conversion will be provided.

This approach will allow Python users to integrate `GenomicPrediction.jl` seamlessly into their existing machine learning pipelines (e.g., with scikit-learn's `GridSearchCV`).
