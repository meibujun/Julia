# Dynamic Orthogonal Epistasis GBLUP in Julia

This Julia package provides a high-performance implementation of a dynamic orthogonal epistasis GBLUP model for genomic prediction, as described in the paper "Dynamic Orthogonal Epistasis Enhances Genomic Prediction in Mongolian Sheep".

## Features

-   **Data Simulation:** Simulates a realistic livestock population with additive and epistatic QTL effects.
-   **GBLUP Models:** Implements both standard additive GBLUP and the advanced orthogonal epistasis GBLUP.
-   **GPU Acceleration:** Leverages CUDA.jl to accelerate the most computationally intensive calculations.
-   **Dynamic Modeling:** Updates model parameters across generations to maintain prediction accuracy under selection.

## Installation

To use this package, you need to have Julia installed. You can then clone this repository and instantiate the project environment:

```bash
git clone <repository-url>
cd DynamicEpistasisGBLUP
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Usage

The main workflow can be run from the Julia REPL or by running the example script.

### Running from the REPL

Start Julia from the project root directory:
```bash
julia --project
```

Then, within the Julia session:
```julia
using DynamicEpistasisGBLUP

# Run the full simulation and analysis.
# This will use the GPU if available.
MainWorkflow.run_full_workflow()

# To force CPU usage:
# MainWorkflow.run_full_workflow(use_gpu=false)
```

### Running the Example Script

You can also run the provided example script from your terminal:
```bash
julia --project examples/run_simulation.jl
```

This will execute the full workflow and save the results plot.

## Workflow Overview

The `run_full_workflow()` function performs the following steps:
1.  **Simulates Data:** Creates a base population and simulates 10 generations of selection for a trait with both additive and epistatic effects.
2.  **Runs Models:** For each generation, it runs two genomic prediction models:
    *   A standard additive GBLUP.
    *   An orthogonal epistasis GBLUP that models both additive and epistatic effects.
3.  **Evaluates Accuracy:** It calculates the Pearson correlation between the true genetic values and the predicted values for both models.
4.  **Generates Plot:** It creates a plot named `prediction_accuracy.png` that visualizes the prediction accuracy of both models over the 10 generations.

## Modules

-   `DynamicEpistasisGBLUP`: The main module that integrates all components.
-   `Simulation`: Handles the population simulation.
-   `GBLUP`: Contains the core logic for the genomic prediction models.
-   `GPUAcceleration`: Provides GPU-accelerated functions for matrix calculations.
-   `MainWorkflow`: Orchestrates the overall simulation and analysis.
