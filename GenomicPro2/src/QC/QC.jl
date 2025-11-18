"""
# QC Module

Quality control functions for genomic data.

Implements standard QC procedures:
- Minor allele frequency (MAF) filtering
- Missing rate filtering (samples and markers)
- Hardy-Weinberg Equilibrium (HWE) testing
- Call rate filtering
- Sample duplicate detection
- Linkage Disequilibrium (LD) pruning

## Examples

```julia
# Apply standard QC filters
geno_qc = quality_control(geno;
    min_maf = 0.01,
    max_missing_per_marker = 0.1,
    max_missing_per_sample = 0.1,
    hwe_pvalue = 1e-6
)

# Get QC report
report = qc_report(geno)
println(report)
```
"""
module QC

using LinearAlgebra
using Statistics
using Printf

using ..Core
using ..Data

# Export QC functions
export quality_control, qc_report
export filter_maf, filter_missing, filter_hwe
export hardy_weinberg_test, call_rate
export identify_duplicates, compute_sample_correlation
export QCReport, QCFilters

# Export LD pruning functions
export LDResult, LDMatrix
export compute_ld_r2, compute_ld_dprime, compute_ld_full
export ld_prune_window, ld_prune_pairwise
export compute_ld_matrix

include("filters.jl")
include("statistics.jl")
include("reports.jl")
include("ld_pruning.jl")

end # module QC
