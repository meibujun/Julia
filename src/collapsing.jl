# src/collapsing.jl

using .RareVariantEpistasis
using StatsBase, Distributions, LinearAlgebra, HypothesisTests

"""
    collapsing_analysis(g::GenomicData, p::PhenotypeData, method::String="CAST")

执行稀有变异的折叠分析。

参数:
- `g::GenomicData`: 基因组数据
- `p::PhenotypeData`: 表型数据
- `method::String`: 折叠方法，可选 "CAST" 或 "WSS"

返回:
- `DataFrame`: 包含每个基因或区域的检验结果
"""
function collapsing_analysis(g::GenomicData, p::PhenotypeData; method::String="CAST")
    if method == "CAST"
        return cast_analysis(g, p)
    elseif method == "WSS"
        return wss_analysis(g, p)
    else
        error("Unsupported collapsing method: $method")
    end
end

"""
    cast_analysis(g::GenomicData, p::PhenotypeData)

执行 CAST (Cohort Allelic Sums Test) 分析。

参数:
- `g::GenomicData`: 基因组数据
- `p::PhenotypeData`: 表型数据

返回:
- `DataFrame`: 包含每个基因或区域的检验结果
"""
function cast_analysis(g::GenomicData, p::PhenotypeData)
    # 识别病例和对照组 (假设 1=case, 0=control)
    cases = p.phenotypes[p.phenotypes.phenotype .== 1, :sample_id]
    controls = p.phenotypes[p.phenotypes.phenotype .== 0, :sample_id]

    case_indices = [i for (i, id) in enumerate(g.sample_ids) if id in cases]
    control_indices = [i for (i, id) in enumerate(g.sample_ids) if id in controls]

    case_genotypes = g.genotypes[case_indices, :]
    control_genotypes = g.genotypes[control_indices, :]

    # 对每个基因或区域进行检验
    # 简化的示例: 将所有SNP视为一个区域

    # 计算病例组和对照组中稀有变异的携带者数量
    case_carriers = sum(vec(sum(case_genotypes, dims=2)) .> 0)
    control_carriers = sum(vec(sum(control_genotypes, dims=2)) .> 0)

    case_non_carriers = length(cases) - case_carriers
    control_non_carriers = length(controls) - control_carriers

    # 构建列联表
    contingency_table = [case_carriers control_carriers; case_non_carriers control_non_carriers]

    # 使用 Fisher's 精确检验
    p_val = pvalue(FisherExactTest(contingency_table...))

    results = DataFrame(gene="all_variants", p_value=p_val)

    return results
end

"""
    wss_analysis(g::GenomicData, p::PhenotypeData)

执行 WSS (Weighted Sum Statistic) 分析。

参数:
- `g::GenomicData`: 基因组数据
- `p::PhenotypeData`: 表型数据

返回:
- `DataFrame`: 包含每个基因或区域的检验结果
"""
function wss_analysis(g::GenomicData, p::PhenotypeData)
    # 识别病例和对照组 (假设 1=case, 0=control)
    cases = p.phenotypes[p.phenotypes.phenotype .== 1, :sample_id]
    controls = p.phenotypes[p.phenotypes.phenotype .== 0, :sample_id]

    case_indices = [i for (i, id) in enumerate(g.sample_ids) if id in cases]
    control_indices = [i for (i, id) in enumerate(g.sample_ids) if id in controls]

    case_genotypes = g.genotypes[case_indices, :]
    control_genotypes = g.genotypes[control_indices, :]

    # 简化的示例: 将所有SNP视为一个区域, 权重为1

    # 计算每个个体的加权和
    case_scores = vec(sum(case_genotypes, dims=2))
    control_scores = vec(sum(control_genotypes, dims=2))

    # 使用 t-检验
    p_val = pvalue(EqualVarianceTTest(case_scores, control_scores))

    results = DataFrame(gene="all_variants", p_value=p_val)

    return results
end
