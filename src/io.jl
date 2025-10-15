# src/io.jl

using .RareVariantEpistasis
using SnpArrays, VariantCallFormat, CSV, DataFrames

"""
    load_plink(bed_path::String, bim_path::String, fam_path::String)

从 PLINK 文件加载基因组数据。

参数:
- `bed_path`: .bed 文件路径
- `bim_path`: .bim 文件路径
- `fam_path`: .fam 文件路径

返回:
- `GenomicData`: 包含基因型、SNP 信息和样本 ID 的 GenomicData 对象
"""
function load_plink(bed_path::String, bim_path::String, fam_path::String)
    # 使用 SnpArrays.jl 包读取数据
    snp_data = SnpArray(bed_path)

    # 读取 .fam 文件获取样本 ID
    fam_data = CSV.read(fam_path, DataFrame, header=false)
    sample_ids = fam_data[!, 2]

    # 读取 .bim 文件获取 SNP 信息
    bim_data = CSV.read(bim_path, DataFrame, header=false)
    snp_info = [SNPInfo(row[1], row[2], row[4], row[5], row[6]) for row in eachrow(bim_data)]

    # 基因型数据
    genotypes = convert(Matrix{Int8}, snp_data)

    return GenomicData(genotypes, snp_info, sample_ids)
end

"""
    load_vcf(vcf_path::String)

从 VCF 文件加载基因组数据。

参数:
- `vcf_path`: .vcf 文件路径

返回:
- `GenomicData`: 包含基因型、SNP 信息和样本 ID 的 GenomicData 对象
"""
function load_vcf(vcf_path::String)
    # 使用 VCF.jl 包读取数据
    reader = VCF.Reader(open(vcf_path, "r"))

    # 获取样本 ID
    sample_ids = VariantCallFormat.header(reader).sampleID

    snp_info = SNPInfo[]
    all_genotypes = []

    for record in reader
        # 提取 SNP 信息
        chr = VCF.chrom(record)
        pos = VCF.pos(record)
        id = VCF.id(record)
        ref = VCF.ref(record)
        alt = VCF.alt(record)
        push!(snp_info, SNPInfo(chr, id[1], pos, ref, alt[1])) # 假设只有一个备选等位基因

        # 提取基因型数据
        push!(all_genotypes, VCF.genotype(record, 1:length(sample_ids), "GT"))
    end

    # 将基因型列表转换为矩阵
    n_samples = length(sample_ids)
    n_snps = length(snp_info)
    genotypes = Matrix{Int8}(undef, n_samples, n_snps)
    for (j, snp_genotypes) in enumerate(all_genotypes)
        for (i, gt) in enumerate(snp_genotypes)
            if gt == "0/0"
                genotypes[i, j] = 0
            elseif gt == "0/1" || gt == "1/0"
                genotypes[i, j] = 1
            elseif gt == "1/1"
                genotypes[i, j] = 2
            else
                genotypes[i, j] = -1 # Missing
            end
        end
    end

    return GenomicData(genotypes, snp_info, sample_ids)
end

"""
    load_csv(csv_path::String; sample_id_col::String, snp_cols::UnitRange{Int})

从 CSV 文件加载基因组数据。

参数:
- `csv_path`: .csv 文件路径
- `sample_id_col`: 样本 ID 所在的列名
- `snp_cols`: SNP 基因型数据所在的列范围

返回:
- `GenomicData`: 包含基因型、SNP 信息和样本 ID 的 GenomicData 对象
"""
function load_csv(csv_path::String; sample_id_col::String, snp_cols::UnitRange{Int})
    df = CSV.read(csv_path, DataFrame)

    sample_ids = df[!, sample_id_col]
    genotypes = Matrix{Int8}(df[!, snp_cols])

    # 对于 CSV，SNP 信息可能不完整，需要用户提供或进行默认处理
    snp_info = [SNPInfo("Unknown", "SNP$i", i, "N", "N") for i in 1:size(genotypes, 2)]

    return GenomicData(genotypes, snp_info, sample_ids)
end
