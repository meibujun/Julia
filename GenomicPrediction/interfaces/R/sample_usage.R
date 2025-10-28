# interfaces/R/sample_usage.R
# -----------------------------
#
# This script demonstrates how to call the GenomicPrediction.jl package from R
# using the JuliaCall library.

# --- 1. Setup ---
# First, ensure you have JuliaCall installed in R:
# install.packages("JuliaCall")

library(JuliaCall)

# --- 2. Initialize Julia ---
cat("Initializing Julia runtime...\n")
# Point to your Julia installation if not in PATH
# julia_setup(JULIA_HOME = "/path/to/julia/bin")
julia_setup()

# --- 3. Activate Project and Load Package ---
cat("Activating Julia project and loading GenomicPrediction.jl...\n")
# Go up two directories to the root of the GenomicPrediction.jl package
julia_command("using Pkg; Pkg.activate('../../')")

# Load the module. We can assign it to an R object for convenience.
julia_library("GenomicPrediction")

# --- 4. Prepare Data ---
cat("\n--- Step 4: Preparing Data ---\n")
# Create dummy data files for the example
geno_path <- "../sample_data/genotypes.csv"
pheno_path <- "../sample_data/phenotypes.csv"
write.csv(data.frame(ID = 1:5, m1 = c(0,1,2,0,1), m2 = c(2,1,0,2,1)), geno_path, row.names = FALSE)
write.csv(data.frame(ID = 1:5, y = c(1.1, 1.9, 3.2, 1.2, 2.3)), pheno_path, row.names = FALSE)

# Call the Julia function to load data
# The 'julia_call' function executes a Julia function with R arguments
mock_data <- julia_call("GenomicPrediction.load_csv", geno_path, pheno_path, header_geno=TRUE, header_pheno=TRUE)
cat("Data loaded successfully into a Julia object.\n")


# --- 5. Initialize and Train Model ---
cat("\n--- Step 5: Training GBLUP Model ---\n")
# Create a GBLUP model instance
gblup_model <- julia_call("GenomicPrediction.GBLUPModel", 10.0)

# Train the model
# Note: Julia functions with '!' are mutated. In R, we just call them.
julia_call("GenomicPrediction.fit!", gblup_model, mock_data)
cat("Model training complete.\n")


# --- 6. Save and Load Model ---
cat("\n--- Step 6: Saving and Reloading Model ---\n")
temp_path <- "temp_model.bson"
julia_call("GenomicPrediction.save_model", gblup_model, temp_path)
cat(paste("Model saved to", temp_path, "\n"))

loaded_model <- julia_call("GenomicPrediction.load_model", temp_path)
cat("Model successfully reloaded.\n")

# --- 7. Make Predictions ---
cat("\n--- Step 7: Making Predictions ---\n")
# Extract the genotype DataFrame from the Julia object
genotypes_df <- mock_data$genotypes

# Predict using the reloaded model
predictions_jl <- julia_call("GenomicPrediction.predict", loaded_model, genotypes_df)

# Convert Julia vector to R vector
predictions_r <- julia_eval("x -> collect(x)", predictions_jl)

cat("Predictions obtained and converted to R vector.\n")

# --- 8. Display Results ---
cat("\n--- Step 8: Results ---\n")
cat("Predicted values:\n")
print(predictions_r)

# Clean up temporary files
file.remove(geno_path)
file.remove(pheno_path)
file.remove(temp_path)
