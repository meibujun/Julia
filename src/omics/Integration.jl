module Integration

using Random
using Statistics
using LinearAlgebra
using DataFrames
using SparseArrays
using ..MathUtils

"""
    simulate_multiomics(; n_samples = 500, feature_config = nothing, seed = 2025)

构建融合基因组、转录组、蛋白质组、代谢组、表观组与表型/系谱的模拟数据。
返回包含多组学矩阵、真实育种值与环境噪声的结构化字典。
"""
function simulate_multiomics(; n_samples::Int = 500,
                               feature_config = nothing,
                               seed::Int = 2025)
    rng = MersenneTwister(seed)
    default = Dict(
        :genomics => 2000,
        :transcriptomics => 800,
        :proteomics => 400,
        :metabolomics => 300,
        :epigenomics => 600,
        :phenomics => 6
    )
    feature_config = isnothing(feature_config) ? default : merge(default, feature_config)
    geno = simulate_genotype_matrix(n_samples, feature_config[:genomics]; rare_rate = 0.02)
    β = randn(rng, feature_config[:genomics]) .* 0.1
    tbv = geno * β
    noise = randn(rng, n_samples)
    transcript = tbv .* randn(rng, feature_config[:transcriptomics])' .+ 0.1 * randn(rng, n_samples, feature_config[:transcriptomics])
    proteome = tbv .* randn(rng, feature_config[:proteomics])' .+ 0.2 * randn(rng, n_samples, feature_config[:proteomics])
    metabol = tbv .* randn(rng, feature_config[:metabolomics])' .+ 0.3 * randn(rng, n_samples, feature_config[:metabolomics])
    epigenome = randn(rng, n_samples, feature_config[:epigenomics]) .* 0.5 .+ tbv .* randn(rng, feature_config[:epigenomics])'
    phenos = DataFrame(
        Animal = string.(:A, 1:n_samples),
        DailyGain = tbv .+ 0.5 * noise,
        CarcassWeight = tbv .+ 0.4 * noise,
        LitterSize = 1.5 .+ 0.2 * tbv .+ 0.5 * randn(rng, n_samples),
        ConceptionRate = clamp.(0.6 .+ 0.1 * tbv .+ 0.1 * randn(rng, n_samples), 0.0, 1.0),
        FeedEfficiency = 1.2 .- 0.1 * tbv .+ 0.1 * randn(rng, n_samples),
        Resilience = tbv .+ randn(rng, n_samples)
    )
    pedigree = DataFrame(ID = phenos.Animal, Sire = fill("0", n_samples), Dam = fill("0", n_samples))
    return Dict(
        :genomics => geno,
        :transcriptomics => transcript,
        :proteomics => proteome,
        :metabolomics => metabol,
        :epigenomics => epigenome,
        :phenomics => phenos,
        :pedigree => pedigree,
        :tbv => tbv
    )
end

"""
    integrate_multiomics!(datasets; method = :zscore, weights = nothing)

对多组学矩阵进行标准化并融合为统一的综合评分矩阵，支持 z-score、
主成分与互信息权重等策略。
"""
function integrate_multiomics!(datasets::Dict; method::Symbol = :zscore, weights = nothing)
    matrices = [:genomics, :transcriptomics, :proteomics, :metabolomics, :epigenomics]
    weights = isnothing(weights) ? Dict(k => 1.0 for k in matrices) : merge(Dict(k => 1.0 for k in matrices), weights)
    n = size(datasets[:genomics], 1)
    integrated = zeros(Float64, n, length(matrices))
    for (idx, key) in enumerate(matrices)
        mat = datasets[key]
        if method == :zscore
            standardized = (mat .- mean(mat; dims = 1)) ./ (std(mat; dims = 1) .+ eps())
            integrated[:, idx] = vec(mean(standardized; dims = 2)) .* weights[key]
        elseif method == :pca
            Σ = cov(mat)
            vals, vecs = eigen(Symmetric(Σ))
            comp = mat * vecs[:, end]
            integrated[:, idx] = comp .* weights[key]
        elseif method == :mutual
            geno_score = vec(mean(datasets[:genomics]; dims = 2))
            integrated[:, idx] = mutual_weight(geno_score, mat) .* weights[key]
        else
            error("未知的整合方法: $method")
        end
    end
    datasets[:integrated_score] = integrated
    return datasets
end

function mutual_weight(geno_score::AbstractVector, omics::AbstractMatrix)
    n = length(geno_score)
    scores = zeros(Float64, n)
    for i in 1:n
        row = omics[i, :]
        scores[i] = geno_score[i] * mean(row)
    end
    return scores ./ (std(scores) + eps())
end

"""
    omics_kernel(geno, datasets; weights = Dict(:genomics=>0.4, :transcriptomics=>0.2, ...))

根据多组学结果构建可用于 RKHS / EG-BLUP 的综合核矩阵。
"""
function omics_kernel(geno::AbstractMatrix, datasets::Dict;
                      weights = Dict(:genomics => 0.4, :transcriptomics => 0.2, :proteomics => 0.15,
                                      :metabolomics => 0.15, :epigenomics => 0.1))
    n = size(geno, 1)
    K = zeros(Float64, n, n)
    for (key, w) in weights
        mat = datasets[key]
        kernel = mat * mat'
        MathUtils.symmetrize!(kernel)
        K .+= w * kernel / max(tr(kernel), eps())
    end
    return K
end

end # module
