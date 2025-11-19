"""
VCF/BCF File I/O Module

Support for reading and writing Variant Call Format (VCF) files.

VCF is the standard format for storing genetic variation data. This module
provides functions to read VCF files and convert them to CompactGenotypes.

# Format Support
- VCF text files (.vcf)
- Compressed VCF files (.vcf.gz) - requires CodecZlib
- Basic BCF support (future)

# Features
- Fast genotype parsing
- Multi-allelic variant handling
- Missing data support
- Flexible filtering (by sample, variant, quality)
- INFO and FORMAT field parsing

# References
- Danecek et al. (2011). The variant call format and VCFtools. Bioinformatics, 27(15), 2156-2158.
- VCF Specification v4.2+: https://samtools.github.io/hts-specs/VCFv4.2.pdf
"""

using Printf

"""
    VCFHeader

VCF file header information.

# Fields
- `file_format::String`: VCF version (e.g., "VCFv4.2")
- `meta_lines::Vector{String}`: All ##-prefixed header lines
- `contig_info::Dict{String,Int}`: Contig names and lengths
- `info_fields::Dict{String,NamedTuple}`: INFO field definitions
- `format_fields::Dict{String,NamedTuple}`: FORMAT field definitions
- `sample_ids::Vector{String}`: Sample IDs from header
- `n_samples::Int`: Number of samples
"""
struct VCFHeader
    file_format::String
    meta_lines::Vector{String}
    contig_info::Dict{String,Int}
    info_fields::Dict{String,NamedTuple}
    format_fields::Dict{String,NamedTuple}
    sample_ids::Vector{String}
    n_samples::Int
end

function Base.show(io::IO, header::VCFHeader)
    println(io, "VCF Header")
    println(io, "  Format: $(header.file_format)")
    println(io, "  Samples: $(header.n_samples)")
    println(io, "  Contigs: $(length(header.contig_info))")
    println(io, "  INFO fields: $(length(header.info_fields))")
    println(io, "  FORMAT fields: $(length(header.format_fields))")
end

"""
    parse_vcf_header(io::IO) -> VCFHeader

Parse VCF header from an IO stream.

# Arguments
- `io::IO`: Input stream positioned at start of VCF file

# Returns
- `VCFHeader`: Parsed header information
"""
function parse_vcf_header(io::IO)
    file_format = "VCFv4.0"  # Default
    meta_lines = String[]
    contig_info = Dict{String,Int}()
    info_fields = Dict{String,NamedTuple}()
    format_fields = Dict{String,NamedTuple}()
    sample_ids = String[]

    while !eof(io)
        line = readline(io)

        if startswith(line, "##")
            # Meta-information line
            push!(meta_lines, line)

            # Parse specific fields
            if startswith(line, "##fileformat=")
                file_format = replace(line, "##fileformat=" => "")
            elseif startswith(line, "##contig=")
                # Parse contig info: ##contig=<ID=1,length=249250621>
                m = match(r"ID=([^,>]+)", line)
                if !isnothing(m)
                    contig_id = m.captures[1]
                    len_match = match(r"length=(\d+)", line)
                    contig_len = isnothing(len_match) ? 0 : parse(Int, len_match.captures[1])
                    contig_info[contig_id] = contig_len
                end
            elseif startswith(line, "##INFO=")
                # Parse INFO field definition
                # ##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
                m = match(r"ID=([^,>]+)", line)
                if !isnothing(m)
                    field_id = m.captures[1]
                    info_fields[field_id] = (line = line,)  # Store full line
                end
            elseif startswith(line, "##FORMAT=")
                # Parse FORMAT field definition
                m = match(r"ID=([^,>]+)", line)
                if !isnothing(m)
                    field_id = m.captures[1]
                    format_fields[field_id] = (line = line,)
                end
            end

        elseif startswith(line, "#")
            # Column header line
            # #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  sample1 sample2 ...
            parts = split(line, '\t')

            if length(parts) > 9
                # Extract sample IDs (columns after FORMAT)
                sample_ids = String.(parts[10:end])
            end

            break  # Header ends here
        else
            # Data line - shouldn't happen in header parsing
            break
        end
    end

    return VCFHeader(
        file_format,
        meta_lines,
        contig_info,
        info_fields,
        format_fields,
        sample_ids,
        length(sample_ids)
    )
end

"""
    parse_genotype(gt_str::AbstractString) -> Union{Int,Missing}

Parse VCF genotype string to numeric genotype (0, 1, 2, or missing).

# Genotype Encoding
- "0/0" or "0|0" -> 0 (homozygous reference)
- "0/1", "1/0", "0|1", "1|0" -> 1 (heterozygous)
- "1/1" or "1|1" -> 2 (homozygous alternate)
- "./." or ".|." or "." -> missing

# Multi-allelic Handling
For multi-allelic sites (e.g., "1/2", "2/2"):
- Counts total non-reference alleles
- "1/2" -> 2, "2/2" -> 2

# Arguments
- `gt_str::AbstractString`: Genotype string from VCF FORMAT field

# Returns
- `Union{Int,Missing}`: Numeric genotype or missing
"""
function parse_genotype(gt_str::AbstractString)
    # Handle missing
    if gt_str == "." || gt_str == "./." || gt_str == ".|."
        return missing
    end

    # Split by / or |
    if occursin('/', gt_str)
        alleles = split(gt_str, '/')
    elseif occursin('|', gt_str)
        alleles = split(gt_str, '|')
    else
        # Single allele (haploid)
        if gt_str == "."
            return missing
        end
        allele = tryparse(Int, gt_str)
        return isnothing(allele) ? missing : min(allele, 2)
    end

    # Parse alleles
    a1_str = String(alleles[1])
    a2_str = String(alleles[2])

    if a1_str == "." || a2_str == "."
        return missing
    end

    a1 = tryparse(Int, a1_str)
    a2 = tryparse(Int, a2_str)

    if isnothing(a1) || isnothing(a2)
        return missing
    end

    # Count non-reference alleles
    # For bi-allelic: 0/0->0, 0/1->1, 1/1->2
    # For multi-allelic: treat any non-zero as alternate
    genotype = (a1 > 0 ? 1 : 0) + (a2 > 0 ? 1 : 0)

    return genotype
end

"""
    read_vcf(vcf_file::String;
             max_variants::Union{Int,Nothing}=nothing,
             samples::Union{Vector{String},Nothing}=nothing,
             regions::Union{Vector{String},Nothing}=nothing,
             min_qual::Float64=0.0,
             pass_only::Bool=false,
             biallelic_only::Bool=true,
             verbose::Bool=true) -> CompactGenotypes

Read VCF file and convert to CompactGenotypes.

# Arguments
- `vcf_file::String`: Path to VCF file (.vcf or .vcf.gz)
- `max_variants::Union{Int,Nothing}`: Maximum variants to read (default: all)
- `samples::Union{Vector{String},Nothing}`: Sample IDs to include (default: all)
- `regions::Union{Vector{String},Nothing}`: Genomic regions to include (e.g., ["1", "2"])
- `min_qual::Float64`: Minimum variant quality score (default: 0.0)
- `pass_only::Bool`: Only include variants with FILTER=PASS (default: false)
- `biallelic_only::Bool`: Only include bi-allelic variants (default: true)
- `verbose::Bool`: Print progress (default: true)

# Returns
- `CompactGenotypes`: Genotype data

# Example
```julia
# Read all variants
geno = read_vcf("data.vcf.gz")

# Read only chromosome 1, quality > 30
geno = read_vcf("data.vcf.gz"; regions=["1"], min_qual=30.0)

# Read specific samples
geno = read_vcf("data.vcf.gz"; samples=["sample1", "sample2"])
```

# Notes
- Supports both uncompressed (.vcf) and gzip-compressed (.vcf.gz) files
- For .vcf.gz, requires CodecZlib package (optional dependency)
- Multi-allelic sites are converted to dosage format unless biallelic_only=true
- Missing genotypes are handled appropriately
"""
function read_vcf(vcf_file::String;
                 max_variants::Union{Int, Nothing} = nothing,
                 samples::Union{Vector{String}, Nothing} = nothing,
                 regions::Union{Vector{String}, Nothing} = nothing,
                 min_qual::Float64 = 0.0,
                 pass_only::Bool = false,
                 biallelic_only::Bool = true,
                 verbose::Bool = true)

    if !isfile(vcf_file)
        throw(ArgumentError("VCF file not found: $vcf_file"))
    end

    if verbose
        println("\n" * "="^70)
        println("Reading VCF File")
        println("="^70)
        println("  File: $vcf_file")
    end

    # Open file (handle compression)
    io = open_vcf_file(vcf_file)

    try
        # Parse header
        header = parse_vcf_header(io)

        if verbose
            println("  Format: $(header.file_format)")
            println("  Samples in VCF: $(header.n_samples)")
        end

        # Determine which samples to include
        if isnothing(samples)
            selected_samples = header.sample_ids
            sample_indices = collect(1:header.n_samples)
        else
            sample_indices = indexin(samples, header.sample_ids)
            if any(isnothing.(sample_indices))
                missing_samples = samples[findall(isnothing.(sample_indices))]
                @warn "Some samples not found in VCF: $missing_samples"
                sample_indices = filter(!isnothing, sample_indices)
            end
            selected_samples = header.sample_ids[sample_indices]
        end

        n_selected_samples = length(selected_samples)

        if verbose
            println("  Samples to read: $n_selected_samples")
            if !isnothing(regions)
                println("  Regions: $(join(regions, ", "))")
            end
        end

        # Storage for variant data
        variant_chromosomes = String[]
        variant_positions = Int[]
        variant_ids = String[]
        variant_ref_alleles = String[]
        variant_alt_alleles = String[]
        genotype_data = Vector{Vector{Union{Int,Missing}}}()

        # Read variants
        n_read = 0
        n_filtered = 0
        n_multiallelic_skipped = 0

        while !eof(io)
            line = readline(io)

            if isempty(line) || startswith(line, "#")
                continue
            end

            n_read += 1

            # Parse variant line
            parts = split(line, '\t')

            if length(parts) < 8
                continue  # Invalid line
            end

            chrom = String(parts[1])
            pos = tryparse(Int, parts[2])
            id = String(parts[3])
            ref = String(parts[4])
            alt = String(parts[5])
            qual_str = String(parts[6])
            filter = String(parts[7])
            # info = String(parts[8])
            # format_str = length(parts) >= 9 ? String(parts[9]) : ""

            # Apply filters
            # Region filter
            if !isnothing(regions) && !(chrom in regions)
                n_filtered += 1
                continue
            end

            # Quality filter
            if qual_str != "." && qual_str != ""
                qual = tryparse(Float64, qual_str)
                if !isnothing(qual) && qual < min_qual
                    n_filtered += 1
                    continue
                end
            end

            # FILTER field
            if pass_only && filter != "PASS" && filter != "."
                n_filtered += 1
                continue
            end

            # Multi-allelic filter
            if biallelic_only && occursin(',', alt)
                n_multiallelic_skipped += 1
                continue
            end

            # Extract genotypes
            if length(parts) < 10
                continue  # No genotype data
            end

            # Find GT field index in FORMAT
            format_str = length(parts) >= 9 ? String(parts[9]) : ""
            format_fields = split(format_str, ':')
            gt_index = findfirst(=>("GT"), format_fields)

            if isnothing(gt_index)
                continue  # No GT field
            end

            # Parse genotypes for selected samples
            variant_genotypes = Vector{Union{Int,Missing}}(undef, n_selected_samples)

            for (i, sample_idx) in enumerate(sample_indices)
                sample_data_idx = 9 + sample_idx  # 9 fixed columns + sample index

                if sample_data_idx <= length(parts)
                    sample_data = String(parts[sample_data_idx])
                    sample_fields = split(sample_data, ':')

                    if gt_index <= length(sample_fields)
                        gt_str = String(sample_fields[gt_index])
                        variant_genotypes[i] = parse_genotype(gt_str)
                    else
                        variant_genotypes[i] = missing
                    end
                else
                    variant_genotypes[i] = missing
                end
            end

            # Store variant
            push!(variant_chromosomes, chrom)
            push!(variant_positions, isnothing(pos) ? 0 : pos)
            push!(variant_ids, id == "." ? "variant_$(n_read)" : id)
            push!(variant_ref_alleles, ref)
            push!(variant_alt_alleles, occursin(',', alt) ? split(alt, ',')[1] : alt)
            push!(genotype_data, variant_genotypes)

            # Check max variants limit
            if !isnothing(max_variants) && length(variant_ids) >= max_variants
                if verbose
                    println("  Reached max_variants limit: $max_variants")
                end
                break
            end

            # Progress
            if verbose && n_read % 10000 == 0
                @printf("  Read %d variants (%d passed filters)\r", n_read, length(variant_ids))
            end
        end

        if verbose
            println()
            println("="^70)
            println("VCF Reading Summary")
            println("="^70)
            println("  Variants read: $n_read")
            println("  Variants passed: $(length(variant_ids))")
            println("  Filtered out: $n_filtered")
            if biallelic_only
                println("  Multi-allelic skipped: $n_multiallelic_skipped")
            end
            println("="^70)
        end

        # Convert to matrix
        n_variants = length(variant_ids)
        geno_matrix = Matrix{Union{Int,Missing}}(undef, n_selected_samples, n_variants)

        for j in 1:n_variants
            geno_matrix[:, j] = genotype_data[j]
        end

        # Replace missing with mode or 0
        geno_matrix_int = Matrix{Int}(undef, n_selected_samples, n_variants)
        for j in 1:n_variants
            col = geno_matrix[:, j]
            non_missing = col[.!ismissing.(col)]

            if isempty(non_missing)
                # All missing - fill with 0
                geno_matrix_int[:, j] .= 0
            else
                # Impute missing with mode
                mode_val = round(Int, median(non_missing))
                for i in 1:n_selected_samples
                    geno_matrix_int[i, j] = ismissing(col[i]) ? mode_val : col[i]
                end
            end
        end

        # Create CompactGenotypes
        geno = CompactGenotypes(
            geno_matrix_int,
            selected_samples,
            variant_ids;
            chromosome = variant_chromosomes,
            position = variant_positions,
            ref_allele = variant_ref_alleles,
            alt_allele = variant_alt_alleles
        )

        if verbose
            println("\n✓ VCF file loaded successfully")
            println("  Final dataset: $(geno.n_samples) samples × $(geno.n_markers) variants")
        end

        return geno

    finally
        close(io)
    end
end

"""
    open_vcf_file(vcf_file::String) -> IO

Open VCF file, handling both compressed and uncompressed formats.

# Arguments
- `vcf_file::String`: Path to VCF file

# Returns
- `IO`: Input stream
"""
function open_vcf_file(vcf_file::String)
    if endswith(vcf_file, ".gz")
        # Try to load CodecZlib for gzip support
        try
            # Use eval to avoid hard dependency
            @eval using CodecZlib
            return GzipDecompressorStream(open(vcf_file, "r"))
        catch e
            error("Reading .vcf.gz files requires CodecZlib package. " *
                  "Install with: using Pkg; Pkg.add(\"CodecZlib\")")
        end
    else
        return open(vcf_file, "r")
    end
end

"""
    write_vcf(vcf_file::String, geno::CompactGenotypes;
              file_format::String="VCFv4.2",
              source::String="GenomicPro2.jl",
              reference::String="",
              compress::Bool=false,
              verbose::Bool=true)

Write CompactGenotypes to VCF file.

# Arguments
- `vcf_file::String`: Output VCF file path
- `geno::CompactGenotypes`: Genotype data
- `file_format::String`: VCF version (default: "VCFv4.2")
- `source::String`: Source program name (default: "GenomicPro2.jl")
- `reference::String`: Reference genome name (default: "")
- `compress::Bool`: Compress output with gzip (default: false)
- `verbose::Bool`: Print progress (default: true)

# Example
```julia
write_vcf("output.vcf", geno)
write_vcf("output.vcf.gz", geno; compress=true)
```
"""
function write_vcf(vcf_file::String, geno::CompactGenotypes;
                  file_format::String = "VCFv4.2",
                  source::String = "GenomicPro2.jl",
                  reference::String = "",
                  compress::Bool = false,
                  verbose::Bool = true)

    if verbose
        println("\n" * "="^70)
        println("Writing VCF File")
        println("="^70)
        println("  File: $vcf_file")
        println("  Samples: $(geno.n_samples)")
        println("  Variants: $(geno.n_markers)")
    end

    # Open output file
    if compress || endswith(vcf_file, ".gz")
        try
            @eval using CodecZlib
            io = GzipCompressorStream(open(vcf_file, "w"))
        catch e
            error("Writing compressed VCF requires CodecZlib package. " *
                  "Install with: using Pkg; Pkg.add(\"CodecZlib\")")
        end
    else
        io = open(vcf_file, "w")
    end

    try
        # Write header
        println(io, "##fileformat=$file_format")
        println(io, "##source=$source")

        if !isempty(reference)
            println(io, "##reference=$reference")
        end

        # Write FORMAT definitions
        println(io, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")

        # Write column header
        header_line = join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"], '\t')
        header_line *= "\t" * join(geno.sample_ids, '\t')
        println(io, header_line)

        # Write variants
        for i in 1:geno.n_markers
            chrom = geno.chromosome[i]
            pos = geno.position[i]
            id = geno.marker_ids[i]
            ref = geno.ref_allele[i]
            alt = geno.alt_allele[i]

            # Get genotypes for this variant
            genotypes = String[]
            for j in 1:geno.n_samples
                g = get_genotype(geno, j, i)

                if ismissing(g)
                    push!(genotypes, "./.")
                elseif g == 0
                    push!(genotypes, "0/0")
                elseif g == 1
                    push!(genotypes, "0/1")
                elseif g == 2
                    push!(genotypes, "1/1")
                else
                    push!(genotypes, "./.")
                end
            end

            # Write variant line
            variant_line = join([chrom, pos, id, ref, alt, ".", "PASS", ".", "GT"], '\t')
            variant_line *= "\t" * join(genotypes, '\t')
            println(io, variant_line)

            # Progress
            if verbose && i % 10000 == 0
                @printf("  Written %d/%d variants\r", i, geno.n_markers)
            end
        end

        if verbose
            println()
            println("="^70)
            println("✓ VCF file written successfully")
            println("="^70)
        end

    finally
        close(io)
    end
end

# Export
export VCFHeader
export read_vcf, write_vcf
export parse_vcf_header, parse_genotype
