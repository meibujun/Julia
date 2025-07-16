# This is an example script to demonstrate how to use the DynamicEpistasisGBLUP package.

# First, make sure you have the project environment activated.
# You can do this by starting Julia with `julia --project` from the package root directory.

# Import the main module
using DynamicEpistasisGBLUP

println("--- Starting Dynamic Epistasis GBLUP Workflow ---")

# Run the full workflow.
# This function will simulate the population, run the prediction models,
# evaluate their accuracy, and generate a plot.
#
# By default, it will try to use the GPU if a CUDA-enabled device is available.
# To force CPU usage, you can call:
# MainWorkflow.run_full_workflow(use_gpu=false)

MainWorkflow.run_full_workflow()

println("\n--- Workflow Finished ---")
println("Check the generated plot 'prediction_accuracy.png' for the results.")
