# interfaces/R/example.R
#
# Title: Example R Script for Interacting with GenomicPrediction.jl
# Author: Jules
# Date: 2025-10-24
#
# Description:
# This script demonstrates how an R user can leverage the `GenomicPrediction.jl`
# package using the `JuliaCall` R package. It serves as a proof-of-concept
# and a template for building a more formal R wrapper package.
#
# Requirements:
# - R installed
# - The 'JuliaCall' R package installed: install.packages(\"JuliaCall\")
# - Julia (version 1.11.6 as used in the project) installed and accessible.

# --- 1. Setup and Initialization ---

# Load the JuliaCall library
library(JuliaCall)

# Specify the path to your Julia executable.
# JuliaCall will attempt to find it automatically, but explicit is better.
# julia_path <- "/path/to/your/julia-1.11.6/bin/julia"
# julia_setup(JULIA_HOME = dirname(julia_path))

# For this project, we can let it try to find the path.
# The user running this script would need to configure this.
julia_setup()

# Define the absolute path to the GenomicPrediction.jl project root.
# This needs to be set by the user.
project_path <- file.path(getwd(), "GenomicPrediction") # Assuming script is run from repo root

# Activate the Julia project environment. This is crucial for loading
# the correct dependencies.
julia_command(paste0('using Pkg; Pkg.activate("', project_path, '")'))

# Load the GenomicPrediction.jl package into the Julia session
julia_library("GenomicPrediction")

cat("--- Julia Environment and GenomicPrediction.jl Loaded Successfully ---\\n")


# --- 2. Data Loading and Preparation ---

# Load sample data into R data frames.
# Paths are relative to the project root.
geno_path <- file.path(project_path, "test", "sample_data", "genotypes.csv")
pheno_path <- file.path(project_path, "test", "sample_data", "phenotypes.csv")

geno_df <- read.csv(geno_path)
pheno_df <- read.csv(pheno_path)

# The Julia `load_csv` function is convenient, but to demonstrate passing
# data from R to Julia, we'll construct the `GenomicData` object manually.

# Pass the R data frames to the Julia session
julia_assign("r_geno_df", geno_df)
julia_assign("r_pheno_df", pheno_df)

# In Julia, convert them to Julia DataFrames and create the GenomicData object
julia_command("
    using DataFrames;
    jl_geno_df = DataFrame(r_geno_df);
    jl_pheno_df = DataFrame(r_pheno_df, [:ID, :y, :trait2]);
    jl_pheno_df = select(jl_pheno_df, :ID, :y);
    r_data = GenomicData(jl_geno_df, jl_pheno_df);
")

cat("--- Sample Data Prepared in Julia Session ---\\n")


# --- 3. Model Training and Prediction ---

# Define model parameters
lambda_val <- 50.0
julia_assign("lambda", lambda_val)

# Create a GBLUPModel instance in Julia
julia_command("r_model = GBLUPModel(lambda=lambda)")

# Train the model using the non-mutating `fit` function
cat("Training GBLUP model...\\n")
julia_command("trained_r_model <- fit(r_model, r_data)")

# Make predictions
cat("Making predictions...\\n")
julia_command("r_predictions_jl <- predict(trained_r_model, r_data.genotypes)")

# Retrieve the predictions back into R
predictions_r <- julia_eval("r_predictions_jl")


# --- 4. Display Results ---

cat("\\n--- Results ---\\n")
cat("Successfully generated", length(predictions_r), "predictions.\\n")
cat("First 5 predictions:\\n")
print(head(predictions_r, 5))

# The `predictions_r` object is now a standard R vector and can be used
# in any R analysis or plotting library like ggplot2.
cat("\\n--- R Script Finished ---\\n")
