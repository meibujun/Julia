function _stringify(v)
    v === nothing && return ""
    v isa AbstractString && return String(v)
    return string(v)
end

function _to_char(delim)
    delim isa Char && return delim
    delim isa AbstractString && !isempty(delim) && return delim[1]
    throw(ArgumentError("Delimiter must be a single character, got $(delim)"))
end

function _feature_column_index(df::DataFrame, feature_column)
    if feature_column isa Int
        1 <= feature_column <= ncol(df) || throw(ArgumentError("feature_column index out of range"))
        return feature_column
    elseif feature_column isa Symbol
        return findfirst(==(feature_column), propertynames(df)) || throw(ArgumentError("feature column $(feature_column) not found"))
    elseif feature_column isa AbstractString
        return findfirst(==(Symbol(feature_column)), propertynames(df)) || throw(ArgumentError("feature column $(feature_column) not found"))
    else
        throw(ArgumentError("Unsupported feature_column type: $(typeof(feature_column))"))
    end
end

function _parse_float(value)
    if value === missing
        return NaN
    elseif value isa Number
        return Float64(value)
    elseif value isa AbstractString
        stripped = strip(value)
        isempty(stripped) && return NaN
        try
            return parse(Float64, stripped)
        catch err
            throw(ArgumentError("Cannot parse numeric value from '$(value)': $(err)"))
        end
    else
        return Float64(value)
    end
end

function _coerce_to_float_column(column)
    result = Vector{Float64}(undef, length(column))
    for (i, v) in enumerate(column)
        result[i] = _parse_float(v)
    end
    return result
end

function _nanmask(vec::AbstractVector{Float64})
    mask = BitVector(undef, length(vec))
    @inbounds for i in eachindex(vec)
        mask[i] = isnan(vec[i])
    end
    return mask
end

function _nanmean(vec::AbstractVector{Float64})
    total = 0.0
    count = 0
    @inbounds for v in vec
        if !isnan(v)
            total += v
            count += 1
        end
    end
    return count == 0 ? NaN : total / count
end

function _nanmedian(vec::AbstractVector{Float64})
    filtered = filter(!isnan, vec)
    isempty(filtered) && return NaN
    sorted = sort(filtered)
    mid = length(sorted) ÷ 2
    if isodd(length(sorted))
        return sorted[mid + 1]
    else
        return (sorted[mid] + sorted[mid + 1]) / 2
    end
end

function _replace_nan!(vec::AbstractVector{Float64}, value::Float64)
    @inbounds for i in eachindex(vec)
        if isnan(vec[i])
            vec[i] = value
        end
    end
    return vec
end

function _variance(vec::AbstractVector{Float64})
    clean = filter(!isnan, vec)
    length(clean) <= 1 && return 0.0
    μ = mean(clean)
    return sum((x - μ)^2 for x in clean) / (length(clean) - 1)
end

function _center_rows!(matrix::Matrix{Float64})
    for i in axes(matrix, 1)
        row = view(matrix, i, :)
        μ = mean(row)
        row .-= μ
    end
    return matrix
end

function _nan_ratio(vec::AbstractVector{Float64})
    count = count(isnan, vec)
    return length(vec) == 0 ? 0.0 : count / length(vec)
end

function _ensure_weights(datasets::Vector{OmicsDataset}, weights::Dict{String, Float64})
    isempty(weights) && return Dict(ds.name => 1.0 for ds in datasets)
    missing = filter(name -> !haskey(weights, name), getfield.(datasets, :name))
    isempty(missing) || throw(ArgumentError("Missing weights for datasets: $(join(missing, ", "))"))
    return weights
end
