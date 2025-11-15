"""
Custom exception types for GenomicPro2.

All exceptions inherit from `GenomicProException` for easy catching.
"""

# ============================================================================
# Base Exception Type
# ============================================================================

"""
    GenomicProException <: Exception

Base type for all GenomicPro2-specific exceptions.

Catching this type will catch all custom exceptions from the package.

# Example

```julia
try
    # Some operation
catch e
    if e isa GenomicProException
        # Handle GenomicPro-specific error
    else
        # Handle other errors
        rethrow(e)
    end
end
```
"""
abstract type GenomicProException <: Exception end

# ============================================================================
# Specific Exception Types
# ============================================================================

"""
    DataValidationError <: GenomicProException

Thrown when data validation fails.

# Fields

- `msg::String`: Error message
- `field::Symbol`: Field that failed validation
- `value::Any`: The invalid value

# Example

```julia
throw(DataValidationError(
    "Genotype value out of range",
    :genotype,
    5
))
```
"""
struct DataValidationError <: GenomicProException
    msg::String
    field::Symbol
    value::Any
end

function Base.showerror(io::IO, e::DataValidationError)
    print(io, "DataValidationError: ", e.msg)
    print(io, "\n  Field: ", e.field)
    print(io, "\n  Value: ", e.value)
end

"""
    DimensionMismatchError <: GenomicProException

Thrown when matrix/array dimensions don't match expected sizes.

# Fields

- `expected::Tuple`: Expected dimensions
- `actual::Tuple`: Actual dimensions

# Example

```julia
throw(DimensionMismatchError(
    (1000, 50000),
    (1000, 45000)
))
```
"""
struct DimensionMismatchError <: GenomicProException
    expected::Tuple
    actual::Tuple
end

function Base.showerror(io::IO, e::DimensionMismatchError)
    print(io, "DimensionMismatchError: expected dimensions ", e.expected)
    print(io, ", got ", e.actual)
end

"""
    ConvergenceError <: GenomicProException

Thrown when an iterative algorithm fails to converge.

# Fields

- `msg::String`: Error message
- `iterations::Int`: Number of iterations performed
- `residual::Float64`: Final residual value

# Example

```julia
throw(ConvergenceError(
    "PCG solver did not converge",
    1000,
    1e-3
))
```
"""
struct ConvergenceError <: GenomicProException
    msg::String
    iterations::Int
    residual::Float64
end

function Base.showerror(io::IO, e::ConvergenceError)
    print(io, "ConvergenceError: ", e.msg)
    print(io, "\n  Iterations: ", e.iterations)
    print(io, "\n  Final residual: ", e.residual)
end

"""
    FileFormatError <: GenomicProException

Thrown when file format is invalid or not recognized.

# Fields

- `filepath::String`: Path to the problematic file
- `expected_format::String`: Expected file format
- `msg::String`: Error message

# Example

```julia
throw(FileFormatError(
    "data.txt",
    "VCF",
    "File does not have VCF header"
))
```
"""
struct FileFormatError <: GenomicProException
    filepath::String
    expected_format::String
    msg::String
end

function Base.showerror(io::IO, e::FileFormatError)
    print(io, "FileFormatError: ", e.msg)
    print(io, "\n  File: ", e.filepath)
    print(io, "\n  Expected format: ", e.expected_format)
end

"""
    IncompatibleDataError <: GenomicProException

Thrown when two datasets are incompatible (e.g., different sample sets).

# Fields

- `msg::String`: Error message
- `dataset1::String`: Name/ID of first dataset
- `dataset2::String`: Name/ID of second dataset

# Example

```julia
throw(IncompatibleDataError(
    "Sample IDs do not match",
    "genotypes",
    "phenotypes"
))
```
"""
struct IncompatibleDataError <: GenomicProException
    msg::String
    dataset1::String
    dataset2::String
end

function Base.showerror(io::IO, e::IncompatibleDataError)
    print(io, "IncompatibleDataError: ", e.msg)
    print(io, "\n  Dataset 1: ", e.dataset1)
    print(io, "\n  Dataset 2: ", e.dataset2)
end
