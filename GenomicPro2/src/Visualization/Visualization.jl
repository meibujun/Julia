"""
# Visualization Module

Visualization tools for genomic data analysis.

Provides data preparation functions for common genomic plots.
Use with Plots.jl or other plotting backends.

Includes:
- Manhattan plots for GWAS
- QQ plots for p-values
- PCA scatter plots
- LD heatmaps

## Examples

```julia
using GenomicPro2
using Plots

# Manhattan plot
data = manhattan_plot_data(gwas_results)
scatter(data.x_positions, data.minus_log10_p;
        xlabel="Chromosome",
        ylabel="-log₁₀(p)")

# QQ plot
data = qq_plot_data(pvalues)
plot(data.expected, data.observed)

# PCA plot
pca_result = pca(geno)
data = pca_plot_data(pca_result)
scatter(data.pc_x, data.pc_y)
```
"""
module Visualization

using Printf
using Statistics

using ..Core
using ..Data
using ..QC
using ..PopulationStructure

# Include modules
include("plots.jl")

# Export visualization functions
export GWASResult
export manhattan_plot_data, qq_plot_data, pca_plot_data, ld_heatmap_data
export save_plot_data

end # module Visualization
