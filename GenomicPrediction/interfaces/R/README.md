# R Interface for GenomicPrediction.jl

This directory will contain the R interface for the `GenomicPrediction.jl` package.

## Plan

The interface will be provided as a lightweight R package that uses the `JuliaCall` R package to interact with the Julia backend.

### Key Features:
-   A `train_model()` function that takes an R data frame and model parameters, calls the corresponding `fit!` method in Julia, and returns an R object wrapping the trained Julia model.
-   A `predict()` method for the trained model object that takes new data and returns predictions as an R vector or data frame.
-   Utility functions to easily create `GenomicData` objects from R data frames.
-   Conversion of Julia results (like data frames of SNP effects) into native R data structures.

This will allow R users to leverage the high performance of `GenomicPrediction.jl` without leaving their familiar R environment.
