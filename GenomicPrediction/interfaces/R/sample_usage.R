# interfaces/R/sample_usage.R
# -----------------------------
#
# This script demonstrates how to call the GenomicPrediction.jl package from R
# using the JuliaCall library.
#
# Installation:
# 1. Make sure you have Julia installed and accessible in your PATH.
# 2. Install JuliaCall in R: `install.packages("JuliaCall")`
# 3. From R, configure JuliaCall for your environment for the first time:
#    >>> library(JuliaCall)
#    >>> julia_setup()

# --- 1. Setup Julia Environment ---
cat("--- Step 1: Setting up Julia Environment ---\n")
library(JuliaCall)

# By default, JuliaCall will find the Julia executable in your PATH.
# The first call to a Julia function might take a while as it initializes.
julia <- julia_setup(verbose = TRUE)

# Activate the Julia project environment. Assumes the script is run from
# the `GenomicPrediction/interfaces/R` directory.
julia_command("using Pkg; Pkg.activate('../../')")

# --- 2. Load GenomicPrediction.jl and Data ---
cat("\n--- Step 2: Loading Package and Data ---\n")

# Import the main module
julia_library("GenomicPrediction")

# Define file paths
geno_path <- "../sample_data/genotypes.csv"
pheno_path <- "../sample_data/phenotypes.csv"

# Call the `load_csv` function from Julia
# JuliaCall automatically converts R data types to Julia data types.
# Note: In Julia, function names with `!` are not valid in R, so we call them by string.
mock_data <- julia_call("load_csv", geno_path, pheno_path, header_geno=FALSE, header_pheno=TRUE)
cat("Data loaded successfully into a Julia object.\n")

# --- 3. Initialize and Train Model ---
cat("\n--- Step 3: Training GBLUP Model ---\n")

# Initialize the GBLUPModel
model <- julia_call("GBLUPModel", 10.0)

# Train the model using the `fit!` function
# We use julia_call with the function name as a string because `!` is a special char in R
julia_call("fit!", model, mock_data)
cat("Model training complete.\n")


# --- 4. Make Predictions ---
cat("\n--- Step 4: Making Predictions ---\n")

# Extract the genotypes DataFrame from the Julia data object
# The `$` operator can be used to access fields of Julia structs
genotypes_df <- mock_data$genotypes

# Make predictions
predictions_jl <- julia_call("predict", model, genotypes_df)

# The result is a Julia vector. Convert it to an R vector for easier use.
predictions_r <- julia_eval("predictions_jl") # Use julia_eval to get the value
cat("Predictions obtained and converted to R vector.\n")

# --- 5. Display Results ---
cat("\n--- Step 5: Results ---\n")
cat(paste("Number of predictions:", length(predictions_r), "\n"))
cat("Predicted values:\n")
for (i in 1:length(predictions_r)) {
    cat(sprintf("  Individual %d: %.4f\n", i, predictions_r[i]))
}
