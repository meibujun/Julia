module Formats

using CSV
using DataFrames
using DataFrames: Not
using Tables
using CodecZlib
using TranscodingStreams
using Serialization
using LinearAlgebra
using SparseArrays
using ProgressLogging
using StatsBase: mode

const SUPPORTED_EXT = (".vcf", ".vcf.gz", ".bed", ".bim", ".fam", ".csv")

function load_genotypes(path::AbstractString; format::Union{Nothing,Symbol}=nothing, impute::Symbol=:mean)
    fmt = isnothing(format) ? guess_format(path) : format
    if fmt == :vcf
        return load_vcf(path; impute)
    elseif fmt == :plink
        return load_plink(path; impute)
    elseif fmt == :csv
        return load_csv(path; impute)
    else
        error("未知的基因型文件格式: $path")
    end
end

function load_phenotypes(path::AbstractString)
    df = CSV.read(path, DataFrame)
    return df
end

load_covariates(path::AbstractString) = CSV.read(path, DataFrame)

function guess_format(path::AbstractString)
    if endswith(path, ".vcf") || endswith(path, ".vcf.gz")
        return :vcf
    elseif endswith(path, ".bed") || endswith(path, ".bim") || endswith(path, ".fam")
        return :plink
    elseif endswith(path, ".csv")
        return :csv
    else
        error("无法识别的文件扩展名: $path")
    end
end

function load_vcf(path::AbstractString; impute::Symbol=:mean)
    open_stream(path) do io
        samples = String[]
        variants = DataFrame()
        genotype_rows = Vector{Vector{Float64}}()
        for line in eachline(io)
            startswith(line, "##") && continue
            if startswith(line, "#CHROM")
                header = split(line, '\t')
                samples = header[10:end]
                variants = DataFrame(CHROM = String[], POS = Int[], ID = String[], REF = String[], ALT = String[])
                continue
            end
            fields = split(line, '\t')
            push!(variants, (fields[1], parse(Int, fields[2]), fields[3], fields[4], fields[5]))
            format_fields = split(fields[9], ':')
            gt_index = findfirst(==("GT"), format_fields)
            gt_index = isnothing(gt_index) ? 1 : gt_index
            gts = map(x -> parse_gt(x, gt_index), fields[10:end])
            push!(genotype_rows, gts)
        end
        G = reduce(hcat, genotype_rows)'
        impute_missing!(G; method = impute)
        return G, variants, DataFrame(Sample = samples)
    end
end

function load_plink(path::AbstractString; impute::Symbol=:mean)
    prefix = replace(path, ".bed" => "")
    bed_path = string(prefix, ".bed")
    bim_path = string(prefix, ".bim")
    fam_path = string(prefix, ".fam")
    bim = CSV.read(bim_path, DataFrame; header = false, delim = ' ')
    rename!(bim, [:CHR, :SNP, :CM, :POS, :A1, :A2])
    fam = CSV.read(fam_path, DataFrame; header = false, delim = ' ')
    rename!(fam, [:FID, :IID, :PID, :MID, :SEX, :PHENOTYPE])
    n_samples = nrow(fam)
    n_variants = nrow(bim)
    G = Matrix{Float64}(undef, n_samples, n_variants)
    read_bed!(G, bed_path)
    impute_missing!(G; method = impute)
    return G, bim, fam
end

function load_csv(path::AbstractString; impute::Symbol=:mean)
    df = CSV.read(path, DataFrame)
    mat = Matrix{Float64}(select(df, Not(1)))
    impute_missing!(mat; method = impute)
    variants = DataFrame(ID = names(df)[2:end])
    samples = DataFrame(Sample = df[:, 1])
    return mat, variants, samples
end

function open_stream(path::AbstractString, func)
    if endswith(path, ".gz")
        open(path) do fio
            stream = TranscodingStream(GzipDecompressor(), fio)
            return func(stream)
        end
    else
        open(path) do fio
            return func(fio)
        end
    end
end

function parse_gt(gt_field::AbstractString, gt_index::Int)
    parts = split(gt_field, ':')
    gt = parts[gt_index]
    alleles = replace.(split(gt, ['|', '/']), "." => "NaN")
    if any(x -> x == "NaN", alleles)
        return NaN
    end
    return sum(parse.(Float64, alleles))
end

function impute_missing!(G::AbstractMatrix{<:Real}; method::Symbol=:mean)
    for j in axes(G, 2)
        column = view(G, :, j)
        valid = filter(!isnan, column)
        if isempty(valid)
            column .= 0
            continue
        end
        if method == :mean
            μ = mean(valid)
            replace!(column, x -> isnan(x) ? μ : x)
        elseif method == :mode
            μ = mode(valid)
            replace!(column, x -> isnan(x) ? μ : x)
        elseif method == :zero
            replace!(column, x -> isnan(x) ? 0 : x)
        else
            error("未知填补方法: $method")
        end
    end
    return G
end

function read_bed!(dest::AbstractMatrix{Float64}, path::AbstractString)
    open(path, "r") do io
        magic = read(io, NTuple{3,UInt8})
        magic == (0x6C, 0x1B, 0x01) || error("BED 文件格式不兼容")
        n_samples = size(dest, 1)
        n_variants = size(dest, 2)
        bytes_per_variant = cld(n_samples, 4)
        buffer = Vector{UInt8}(undef, bytes_per_variant)
        for j in 1:n_variants
            read!(io, buffer)
            decode_plink_column!(view(dest, :, j), buffer, n_samples)
        end
    end
end

function decode_plink_column!(col::AbstractVector{Float64}, buffer::Vector{UInt8}, n_samples::Int)
    idx = 1
    for byte in buffer
        for shift in (0, 2, 4, 6)
            if idx > n_samples
                return
            end
            bits = (byte >> shift) & 0x03
            col[idx] = bits == 0x00 ? 0.0 : bits == 0x02 ? 1.0 : bits == 0x03 ? 2.0 : NaN
            idx += 1
        end
    end
end

end # module
