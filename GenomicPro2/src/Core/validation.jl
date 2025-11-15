"""
Data validation framework for GenomicPro2.

Provides a flexible system for validating genomic data.
"""

# ============================================================================
# Validation Result
# ============================================================================

"""
    ValidationResult

Result of a data validation operation.

# Fields

- `valid::Bool`: Whether validation passed
- `errors::Vector{String}`: List of error messages
- `warnings::Vector{String}`: List of warning messages
- `metadata::Dict{Symbol, Any}`: Additional validation metadata

# Examples

```julia
result = ValidationResult()
result.valid  # true

push!(result.errors, "Invalid genotype value")
result.valid = false

# Pretty printing
println(result)  # Shows colored output with ✓ or ✗
```
"""
mutable struct ValidationResult
    valid::Bool
    errors::Vector{String}
    warnings::Vector{String}
    metadata::Dict{Symbol, Any}

    function ValidationResult()
        new(true, String[], String[], Dict{Symbol, Any}())
    end
end

"""
    validate(data::AbstractGenomicData) -> ValidationResult

Validate genomic data for consistency and correctness.

This is a generic function that should be implemented by each concrete data type.

# Returns

A `ValidationResult` object containing validation status and any errors/warnings.

# Example

```julia
geno = CompactGenotypes(...)
result = validate(geno)

if !result.valid
    @error "Validation failed" errors=result.errors
end
```
"""
function validate end

# ============================================================================
# ValidationResult Methods
# ============================================================================

"""
    is_valid(result::ValidationResult) -> Bool

Check if validation result indicates valid data.
"""
is_valid(result::ValidationResult) = result.valid

"""
    add_error!(result::ValidationResult, msg::String)

Add an error message and mark validation as failed.
"""
function add_error!(result::ValidationResult, msg::String)
    push!(result.errors, msg)
    result.valid = false
    return result
end

"""
    add_warning!(result::ValidationResult, msg::String)

Add a warning message (does not fail validation).
"""
function add_warning!(result::ValidationResult, msg::String)
    push!(result.warnings, msg)
    return result
end

"""
    merge(results::Vector{ValidationResult}) -> ValidationResult

Merge multiple validation results into one.

The merged result is valid only if all input results are valid.
"""
function Base.merge(results::Vector{ValidationResult})
    merged = ValidationResult()

    merged.valid = all(r -> r.valid, results)
    merged.errors = vcat([r.errors for r in results]...)
    merged.warnings = vcat([r.warnings for r in results]...)

    # Merge metadata
    for result in results
        merge!(merged.metadata, result.metadata)
    end

    return merged
end

"""
Pretty-print validation result with colored output.
"""
function Base.show(io::IO, vr::ValidationResult)
    if get(io, :color, false)
        if vr.valid
            printstyled(io, "✓ Validation PASSED\n"; color=:green, bold=true)
        else
            printstyled(io, "✗ Validation FAILED\n"; color=:red, bold=true)
        end
    else
        if vr.valid
            println(io, "✓ Validation PASSED")
        else
            println(io, "✗ Validation FAILED")
        end
    end

    # Print errors
    if !isempty(vr.errors)
        if get(io, :color, false)
            printstyled(io, "\nErrors ($(length(vr.errors))):\n"; color=:red)
        else
            println(io, "\nErrors ($(length(vr.errors))):")
        end

        for (i, err) in enumerate(vr.errors)
            println(io, "  $i. ", err)
        end
    end

    # Print warnings
    if !isempty(vr.warnings)
        if get(io, :color, false)
            printstyled(io, "\nWarnings ($(length(vr.warnings))):\n"; color=:yellow)
        else
            println(io, "\nWarnings ($(length(vr.warnings))):")
        end

        for (i, warn) in enumerate(vr.warnings)
            println(io, "  $i. ", warn)
        end
    end

    # Print metadata summary
    if !isempty(vr.metadata)
        println(io, "\nMetadata:")
        for (k, v) in vr.metadata
            println(io, "  ", k, ": ", v)
        end
    end
end

# ============================================================================
# Helper Validation Functions
# ============================================================================

"""
    validate_sample_ids(ids::Vector{String}) -> ValidationResult

Validate sample IDs for uniqueness and validity.
"""
function validate_sample_ids(ids::Vector{String})
    result = ValidationResult()

    # Check for empty IDs
    if any(isempty, ids)
        add_error!(result, "Found empty sample IDs")
    end

    # Check for uniqueness
    if length(unique(ids)) != length(ids)
        add_error!(result, "Sample IDs are not unique")

        # Find duplicates
        seen = Set{String}()
        duplicates = String[]
        for id in ids
            if id in seen && !(id in duplicates)
                push!(duplicates, id)
            end
            push!(seen, id)
        end

        result.metadata[:duplicate_ids] = duplicates
    end

    result.metadata[:n_samples] = length(ids)

    return result
end

"""
    validate_dimensions(expected::Tuple, actual::Tuple) -> ValidationResult

Validate that dimensions match expected values.
"""
function validate_dimensions(expected::Tuple, actual::Tuple)
    result = ValidationResult()

    if expected != actual
        add_error!(result,
            "Dimension mismatch: expected $expected, got $actual")
    end

    return result
end

"""
    validate_missing_rate(missing_rate::Float64, max_rate::Float64) -> ValidationResult

Validate that missing rate is within acceptable bounds.
"""
function validate_missing_rate(missing_rate::Float64, max_rate::Float64)
    result = ValidationResult()

    result.metadata[:missing_rate] = missing_rate

    if missing_rate > max_rate
        add_error!(result,
            @sprintf("Missing rate %.2f%% exceeds maximum %.2f%%",
                     missing_rate * 100, max_rate * 100))
    elseif missing_rate > 0.1
        add_warning!(result,
            @sprintf("Missing rate is %.2f%%", missing_rate * 100))
    end

    return result
end
