#!/bin/bash
#
# Slurm Submission Script Template for GenomicPrediction.jl
# =========================================================
#
# This script provides a basic template for running a GenomicPrediction.jl
# analysis on an HPC cluster using the Slurm workload manager.
#
# Usage: sbatch run_on_slurm.sh
#
# You will need to customize the paths and the Julia command.

# --- Slurm Configuration ---
#SBATCH --job-name=GenomicPrediction     # Job name
#SBATCH --output=genpred_job_%j.out      # Standard output log
#SBATCH --error=genpred_job_%j.err       # Standard error log
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (usually 1)
#SBATCH --cpus-per-task=8                # Number of CPU cores requested
#SBATCH --mem=16G                        # Memory requested
#SBATCH --time=01:00:00                  # Time limit

# --- User Configuration ---
# IMPORTANT: Update these paths to match your cluster's environment
JULIA_EXECUTABLE="/path/to/your/julia" # e.g., /opt/apps/julia/1.11.6/bin/julia
PROJECT_DIR="/path/to/your/GenomicPrediction.jl" # The root directory of the package

# --- Environment Setup ---
echo "======================================================"
echo "Job Started: $(date)"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "======================================================"

# Set Julia to use the number of threads requested from Slurm
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- Execution ---
# The command below runs a Julia one-liner.
#   --project=${PROJECT_DIR} : Activates the package environment.
#   -e '...' : Executes the string as Julia code.
#
# This example loads the package, trains a GBLUP model, and prints a success message.
# You should replace this with your actual analysis script or commands.

${JULIA_EXECUTABLE} --project=${PROJECT_DIR} -e '
    using Pkg;
    Pkg.instantiate(); # Ensure all dependencies are installed

    using GenomicPrediction;
    using DataFrames;

    println("--- [1/4] Loading data ---");
    # These paths should be accessible from the compute node
    geno_path = "path/to/your/genotypes.csv";
    pheno_path = "path/to/your/phenotypes.csv";
    # data = load_csv(geno_path, pheno_path);

    println("--- [2/4] Initializing model ---");
    # model = GBLUPModel(10.0);

    println("--- [3/4] Fitting model (Placeholder) ---");
    # fit!(model, data);

    println("--- [4/4] Analysis complete ---");
    println("Example execution finished successfully.");
'

echo "======================================================"
echo "Job Finished: $(date)"
echo "======================================================"
