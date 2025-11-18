"""
# Utils Module

Utility functions for data exploration, validation, and analysis.

Includes:
- Data summary and statistics
- Quality metrics and reporting
- Outlier detection
- Data validation

## Examples

```julia
# Summarize genotype data
summary = summarize(geno)
println(summary)

# Compare two datasets
compare_datasets(geno1, geno2)

# Detect outliers
outliers = detect_outliers(geno; method=:iqr)
```
"""
module Utils

using Statistics
using Printf

using ..Core
using ..Data
using ..QC

# Include utility modules
include("summary.jl")

# Export utilities
export GenotypeDataSummary, PhenotypeDataSummary
export summarize, compare_datasets, detect_outliers, marker_quality_summary

end # module Utils
