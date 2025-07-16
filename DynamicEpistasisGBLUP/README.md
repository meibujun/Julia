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

The main workflow can be run from within a Julia session.

```julia
using DynamicEpistasisGBLUP

# Run the full simulation and analysis
MainWorkflow.run_full_workflow()
```

This will:
1.  Simulate a sheep population for 10 generations.
2.  Run both the additive and epistasis GBLUP models for each generation.
3.  Calculate and print the prediction accuracy for each model.
4.  Generate a plot `prediction_accuracy.png` showing the accuracy trends over generations.

## Modules

-   `DynamicEpistasisGBLUP`: The main module that integrates all components.
-   `Simulation`: Handles the population simulation.
-   `GBLUP`: Contains the core logic for the genomic prediction models.
-   `GPUAcceleration`: Provides GPU-accelerated functions for matrix calculations.
-   `MainWorkflow`: Orchestrates the overall simulation and analysis.
