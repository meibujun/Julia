"""
Visualization Tools for Genomic Data

Provides plotting functions for common genomic visualizations.

Requires Plots.jl to be installed separately.

# Features
- Manhattan plots for GWAS results
- QQ plots for p-value distributions
- PCA scatter plots
- LD heatmaps

# Example
```julia
using Plots

# Manhattan plot
p = manhattan_plot(gwas_results)
savefig(p, "manhattan.png")

# QQ plot
p = qq_plot(pvalues)
savefig(p, "qq.png")
```
"""

using Printf
using Statistics

"""
    GWASResult

Container for GWAS analysis results.

# Fields
- `marker_ids::Vector{String}`: SNP identifiers
- `chromosome::Vector{String}`: Chromosome names
- `position::Vector{Int}`: Base-pair positions
- `pvalues::Vector{Float64}`: P-values from association test
- `effect_sizes::Union{Vector{Float64},Nothing}`: Effect sizes (optional)
- `se::Union{Vector{Float64},Nothing}`: Standard errors (optional)
"""
struct GWASResult
    marker_ids::Vector{String}
    chromosome::Vector{String}
    position::Vector{Int}
    pvalues::Vector{Float64}
    effect_sizes::Union{Vector{Float64},Nothing}
    se::Union{Vector{Float64},Nothing}
end

"""
    manhattan_plot_data(gwas::GWASResult; significance_threshold::Float64=5e-8)

Prepare data for Manhattan plot.

# Arguments
- `gwas::GWASResult`: GWAS results
- `significance_threshold::Float64`: Genome-wide significance threshold (default: 5e-8)

# Returns
NamedTuple with plot data and parameters

# Note
This function prepares the data. Use a plotting backend (Plots.jl recommended)
to create the actual visualization.

# Example
```julia
using Plots

data = manhattan_plot_data(gwas)

# Create plot with Plots.jl
scatter(data.x_positions, data.minus_log10_p;
        group=data.chr_indices,
        xlabel="Chromosome",
        ylabel="-log₁₀(p-value)",
        legend=false)
```
"""
function manhattan_plot_data(gwas::GWASResult; significance_threshold::Float64=5e-8)
    # Calculate -log10(p-values)
    minus_log10_p = -log10.(gwas.pvalues)

    # Get unique chromosomes
    unique_chrs = unique(gwas.chromosome)
    sort!(unique_chrs, lt=(a,b)->begin
        # Try to parse as integers for proper sorting
        ia = tryparse(Int, a)
        ib = tryparse(Int, b)
        if !isnothing(ia) && !isnothing(ib)
            return ia < ib
        else
            return a < b
        end
    end)

    # Assign numeric indices to chromosomes
    chr_to_idx = Dict(chr => i for (i, chr) in enumerate(unique_chrs))
    chr_indices = [chr_to_idx[chr] for chr in gwas.chromosome]

    # Calculate cumulative positions for x-axis
    chr_max_pos = Dict{String,Int}()
    chr_offset = Dict{String,Int}()

    current_offset = 0
    for chr in unique_chrs
        chr_markers = gwas.chromosome .== chr
        if any(chr_markers)
            max_pos = maximum(gwas.position[chr_markers])
            chr_max_pos[chr] = max_pos
            chr_offset[chr] = current_offset
            current_offset += max_pos + 1_000_000  # 1Mb spacing between chromosomes
        end
    end

    # Calculate x positions
    x_positions = zeros(length(gwas.position))
    for i in 1:length(gwas.position)
        chr = gwas.chromosome[i]
        x_positions[i] = chr_offset[chr] + gwas.position[i]
    end

    # Calculate chromosome midpoints for x-axis labels
    chr_labels = String[]
    chr_label_positions = Float64[]

    for chr in unique_chrs
        chr_markers = gwas.chromosome .== chr
        if any(chr_markers)
            chr_x = x_positions[chr_markers]
            mid_pos = (minimum(chr_x) + maximum(chr_x)) / 2
            push!(chr_labels, chr)
            push!(chr_label_positions, mid_pos)
        end
    end

    # Significance threshold
    sig_threshold = -log10(significance_threshold)

    # Identify significant SNPs
    significant = minus_log10_p .> sig_threshold

    return (
        x_positions = x_positions,
        minus_log10_p = minus_log10_p,
        chr_indices = chr_indices,
        chr_labels = chr_labels,
        chr_label_positions = chr_label_positions,
        sig_threshold = sig_threshold,
        significant = significant,
        marker_ids = gwas.marker_ids
    )
end

"""
    qq_plot_data(pvalues::Vector{Float64}; confidence_level::Float64=0.95)

Prepare data for QQ plot of p-values.

# Arguments
- `pvalues::Vector{Float64}`: Observed p-values
- `confidence_level::Float64`: Confidence level for CI (default: 0.95)

# Returns
NamedTuple with plot data

# Example
```julia
using Plots

data = qq_plot_data(pvalues)

# Create QQ plot
plot(data.expected, data.observed;
     seriestype=:scatter,
     xlabel="Expected -log₁₀(p)",
     ylabel="Observed -log₁₀(p)")
plot!(data.expected, data.expected; linestyle=:dash)  # Identity line
```
"""
function qq_plot_data(pvalues::Vector{Float64}; confidence_level::Float64=0.95)
    # Remove NaN and invalid p-values
    valid_pvalues = filter(p -> !isnan(p) && p > 0.0 && p <= 1.0, pvalues)

    if isempty(valid_pvalues)
        throw(ArgumentError("No valid p-values"))
    end

    n = length(valid_pvalues)

    # Sort observed p-values
    sorted_observed = sort(valid_pvalues)

    # Expected p-values under null hypothesis
    expected = [(i - 0.5) / n for i in 1:n]

    # Convert to -log10 scale
    observed_log = -log10.(sorted_observed)
    expected_log = -log10.(expected)

    # Calculate confidence intervals (simplified approximation)
    α = 1 - confidence_level
    ci_lower = zeros(n)
    ci_upper = zeros(n)

    # Simple approximation using percentiles
    for i in 1:n
        # Approximate CI based on order statistics
        # Using ±sqrt(variance) approximation
        rank = i / (n + 1)
        variance = rank * (1 - rank) / (n + 2)
        std_dev = sqrt(variance)

        lower_rank = max(rank - 2 * std_dev, 1e-10)
        upper_rank = min(rank + 2 * std_dev, 1 - 1e-10)

        ci_lower[i] = -log10(lower_rank)
        ci_upper[i] = -log10(upper_rank)
    end

    # Calculate lambda (genomic inflation factor)
    # Lambda = median(chi2) / 0.456 (0.456 is the median of chi2(1))
    median_observed = median(sorted_observed)
    # Approximate chi2 from p-value: chi2 ≈ -2*log(p) for small p
    median_chi2 = -2 * log(median_observed)
    lambda = median_chi2 / 0.456

    return (
        expected = expected_log,
        observed = observed_log,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        lambda = lambda,
        n = n
    )
end

"""
    pca_plot_data(pca_result::PCAResult; pc_x::Int=1, pc_y::Int=2)

Prepare data for PCA scatter plot.

# Arguments
- `pca_result::PCAResult`: PCA results
- `pc_x::Int`: PC for x-axis (default: 1)
- `pc_y::Int`: PC for y-axis (default: 2)

# Returns
NamedTuple with plot data

# Example
```julia
using Plots

data = pca_plot_data(pca_result)

scatter(data.pc_x, data.pc_y;
        xlabel=data.xlabel,
        ylabel=data.ylabel,
        legend=false)
```
"""
function pca_plot_data(pca_result::PCAResult; pc_x::Int=1, pc_y::Int=2)
    if pc_x > pca_result.n_pcs || pc_y > pca_result.n_pcs
        throw(ArgumentError("PC indices exceed available PCs"))
    end

    pc_x_values = pca_result.pcs[:, pc_x]
    pc_y_values = pca_result.pcs[:, pc_y]

    var_x = pca_result.variance_explained[pc_x] * 100
    var_y = pca_result.variance_explained[pc_y] * 100

    xlabel = @sprintf("PC%d (%.2f%%)", pc_x, var_x)
    ylabel = @sprintf("PC%d (%.2f%%)", pc_y, var_y)

    return (
        pc_x = pc_x_values,
        pc_y = pc_y_values,
        xlabel = xlabel,
        ylabel = ylabel,
        sample_ids = pca_result.sample_ids
    )
end

"""
    ld_heatmap_data(geno::CompactGenotypes, marker_indices::Vector{Int})

Prepare LD heatmap data.

# Arguments
- `geno::CompactGenotypes`: Genotype data
- `marker_indices::Vector{Int}`: Indices of markers to include

# Returns
NamedTuple with LD matrix and marker information

# Example
```julia
using Plots

# Get LD for markers 1-50
data = ld_heatmap_data(geno, 1:50)

heatmap(data.r2;
        xlabel="Marker",
        ylabel="Marker",
        title="LD Heatmap")
```
"""
function ld_heatmap_data(geno::CompactGenotypes, marker_indices::Vector{Int})
    ld_matrix = compute_ld_matrix(geno, marker_indices)

    return (
        r2 = ld_matrix.r2,
        r = ld_matrix.r,
        marker_ids = ld_matrix.marker_ids,
        n_markers = length(marker_indices)
    )
end

"""
    save_plot_data(filename::String, data::NamedTuple; format::Symbol=:csv)

Save plot data to file for external plotting.

# Arguments
- `filename::String`: Output filename
- `data::NamedTuple`: Plot data
- `format::Symbol`: Output format (:csv or :tsv)
"""
function save_plot_data(filename::String, data::NamedTuple; format::Symbol=:csv)
    delim = format == :tsv ? '\t' : ','

    open(filename, "w") do io
        # Write header
        header = join(string.(keys(data)), delim)
        println(io, header)

        # Write data
        n_rows = length(data[1])
        for i in 1:n_rows
            row = [string(data[k][i]) for k in keys(data)]
            println(io, join(row, delim))
        end
    end
end

# Export
export GWASResult
export manhattan_plot_data, qq_plot_data, pca_plot_data, ld_heatmap_data
export save_plot_data
