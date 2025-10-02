module AnimalBreeding

using LinearAlgebra
using Statistics
using Random
using SparseArrays

"""
    Pedigree

高性能系谱数据结构，内部存储经过拓扑排序的个体、父母索引以及族谱深度。
所有父母索引满足 `parent_index < individual_index`，以便快速构建关系矩阵。
"""
struct Pedigree
    ids::Vector{String}
    sire::Vector{Int}
    dam::Vector{Int}
    depth::Vector{Int}
    indexmap::Dict{String,Int}
end

"""
    pedigree(ids, sire, dam; check_cycles::Bool = true)

根据个体编号和父母编号构建 `Pedigree` 对象，自动完成：
1. 统一编号类型（全部转换为字符串，支持整数/字符串/缺失值输入）。
2. 族谱拓扑排序，确保父母在子代之前出现。
3. 检测循环系谱并给出友好错误提示。
"""
function pedigree(ids_in::AbstractVector, sire_in::AbstractVector, dam_in::AbstractVector; check_cycles::Bool = true)
    length(ids_in) == length(sire_in) == length(dam_in) || throw(ArgumentError("系谱向量长度必须一致"))
    n = length(ids_in)
    ids = String[id === missing ? "missing_$i" : string(id) for (i, id) in enumerate(ids_in)]
    idmap = Dict{String,Int}(ids[i] => i for i in 1:n)

    sire_idx_raw = _normalise_parent_indices(sire_in, idmap)
    dam_idx_raw = _normalise_parent_indices(dam_in, idmap)

    depth = _compute_pedigree_depths(sire_idx_raw, dam_idx_raw; check_cycles)
    order = sortperm(depth)

    ids_sorted = ids[order]
    invorder = Dict(order[i] => i for i in 1:n)
    sire_sorted = similar(sire_idx_raw)
    dam_sorted = similar(dam_idx_raw)
    depth_sorted = similar(depth)
    for (newpos, oldpos) in enumerate(order)
        sire_sorted[newpos] = _remap_parent(sire_idx_raw[oldpos], invorder)
        dam_sorted[newpos] = _remap_parent(dam_idx_raw[oldpos], invorder)
        depth_sorted[newpos] = depth[oldpos]
    end

    indexmap = Dict{String,Int}(ids_sorted[i] => i for i in 1:n)
    return Pedigree(ids_sorted, sire_sorted, dam_sorted, depth_sorted, indexmap)
end

# ------------------------------ 内部辅助函数 ------------------------------

"""将父母编号转换为内部索引。缺失、空字符串、0 均视为未知父母。"""
function _normalise_parent_indices(parents, idmap)
    idx = Vector{Int}(undef, length(parents))
    for (i, parent) in enumerate(parents)
        if parent === missing || parent === nothing
            idx[i] = 0
        elseif parent isa Integer
            parent == 0 && (idx[i] = 0; continue)
            idx[i] = get(idmap, string(parent), 0)
        else
            str = strip(string(parent))
            isempty(str) && (idx[i] = 0; continue)
            idx[i] = get(idmap, str, 0)
        end
    end
    return idx
end

function _compute_pedigree_depths(sire_idx::Vector{Int}, dam_idx::Vector{Int}; check_cycles::Bool)
    n = length(sire_idx)
    depth_cache = Dict{Int,Int}()
    visited = falses(n)
    stack = falses(n)

    function depth(i)
        visited[i] && return depth_cache[i]
        if stack[i]
            check_cycles || return 0
            error("检测到系谱循环：个体 $(i) 的父母关系存在环")
        end
        stack[i] = true
        sire = sire_idx[i]
        dam = dam_idx[i]
        ds = sire > 0 ? depth(sire) + 1 : 0
        dd = dam > 0 ? depth(dam) + 1 : 0
        d = max(ds, dd)
        depth_cache[i] = d
        stack[i] = false
        visited[i] = true
        return d
    end

    result = Vector{Int}(undef, n)
    for i in 1:n
        result[i] = depth(i)
    end
    return result
end

_remap_parent(parent::Int, invorder::Dict{Int,Int}) = parent == 0 ? 0 : invorder[parent]

# ------------------------------ 关系矩阵构建 ------------------------------

"""
    numerator_relationship(ped::Pedigree; sparse::Bool = false)

基于 Henderson 递推公式构建 A 矩阵（分子关系矩阵）。
可选择返回稠密矩阵或稀疏矩阵，满足大规模种畜评估需求。
"""
function numerator_relationship(ped::Pedigree; sparse::Bool = false)
    n = length(ped.ids)
    A = Matrix{Float64}(undef, n, n)
    fill!(A, 0.0)
    for i in 1:n
        sire = ped.sire[i]
        dam = ped.dam[i]
        if sire == 0 && dam == 0
            A[i, i] = 1.0
        elseif sire == 0 || dam == 0
            parent = sire > 0 ? sire : dam
            A[i, i] = 1.0 + 0.5 * A[parent, parent]
            @inbounds for j in 1:i-1
                A[i, j] = 0.5 * A[parent, j]
                A[j, i] = A[i, j]
            end
        else
            A[i, i] = 1.0 + 0.5 * A[sire, dam]
            @inbounds for j in 1:i-1
                A[i, j] = 0.5 * (A[sire, j] + A[dam, j])
                A[j, i] = A[i, j]
            end
        end
    end
    return sparse ? sparse(A) : A
end

"""
    inverse_numerator_relationship(ped::Pedigree)

使用 Henderson (1976) 算法快速构建 A⁻¹，返回对称稀疏矩阵。
该实现适合后续方程组求解或惩罚项构造。
"""
function inverse_numerator_relationship(ped::Pedigree)
    n = length(ped.ids)
    acc = Dict{Tuple{Int,Int},Float64}()

    add!(i::Int, j::Int, v::Float64) = begin
        key = i <= j ? (i, j) : (j, i)
        acc[key] = get(acc, key, 0.0) + v
    end

    for i in 1:n
        sire = ped.sire[i]
        dam = ped.dam[i]
        if sire == 0 && dam == 0
            add!(i, i, 1.0)
        elseif sire == 0 || dam == 0
            parent = sire > 0 ? sire : dam
            add!(i, i, 1.5)
            add!(parent, parent, 0.5)
            add!(i, parent, -0.5)
        else
            add!(i, i, 2.0)
            add!(sire, sire, 0.5)
            add!(dam, dam, 0.5)
            add!(sire, dam, 0.5)
            add!(i, sire, -1.0)
            add!(i, dam, -1.0)
        end
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for ((r, c), v) in acc
        push!(rows, r); push!(cols, c); push!(vals, v)
        if r != c
            push!(rows, c); push!(cols, r); push!(vals, v)
        end
    end
    return sparse(rows, cols, vals, n, n)
end

# ------------------------------ 基因组矩阵 ------------------------------

"""
    genomic_relationship(M; method::Symbol = :VanRaden, regularization::Float64 = 0.0,
                         allelefreq::Union{Nothing,AbstractVector{<:Real}} = nothing)

使用 VanRaden 等方法计算基因组关系矩阵 G。`M` 为 0/1/2 编码的基因型矩阵（个体×标记）。
自动估计等位基因频率并执行标准化，可选岭回归调节项以改善病态情况。
"""
function genomic_relationship(M::AbstractMatrix{<:Real}; method::Symbol = :VanRaden,
        regularization::Float64 = 0.0, allelefreq::Union{Nothing,AbstractVector{<:Real}} = nothing)
    n, m = size(M)
    m > 0 || throw(ArgumentError("基因型矩阵至少包含一个标记"))
    freq = allelefreq === nothing ? vec(mean(M; dims = 1)) ./ 2 : Float64.(allelefreq)  # freq in [0,1]
    length(freq) == m || throw(ArgumentError("等位基因频率长度必须与标记数量一致"))

    # 中心化矩阵 Z = M - 2p
    Z = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:m
        pj = freq[j]
        col = view(Z, :, j)
        @inbounds for i in 1:n
            col[i] = float(M[i, j]) - 2pj
        end
    end

    denominator = 2sum(freq .* (1 .- freq))
    denominator > 0 || throw(ArgumentError("等位基因频率导致分母为零，请检查输入"))

    G = method === :VanRaden ? (Z * transpose(Z)) / denominator : _alternative_genomic(Z, method, denominator)
    if regularization > 0
        G = (1 - regularization) * G + regularization * I
    end
    return Symmetric(G)
end

function _alternative_genomic(Z::AbstractMatrix{<:Real}, method::Symbol, denominator::Float64)
    if method === :Yang
        n = size(Z, 1)
        return Symmetric((Z * transpose(Z)) / (denominator * n))
    else
        throw(ArgumentError("暂不支持的方法：$(method)。当前仅支持 :VanRaden 或 :Yang"))
    end
end

"""
    hybrid_relationship(ped::Pedigree, G::AbstractMatrix;
                        tau::Real = 0.95, omega::Real = 0.05,
                        genotyped_indices::AbstractVector{<:Integer} = collect(1:size(G, 1)))

构建单步法所需的 H 矩阵（Aguilar et al., 2010）。
`tau` 控制 G 与 A₂₂ 的尺度匹配，`omega` 为针对病态矩阵的轻微调节，
`genotyped_indices` 指定 G 矩阵对应的系谱个体位置，可用于乱序或子集情况。
"""
function hybrid_relationship(ped::Pedigree, G::AbstractMatrix; tau::Real = 0.95, omega::Real = 0.05,
        genotyped_indices::AbstractVector{<:Integer} = collect(1:size(G, 1)))
    n = length(ped.ids)
    size(G, 1) == size(G, 2) || throw(ArgumentError("G 必须为方阵"))
    length(genotyped_indices) == size(G, 1) || throw(ArgumentError("基因型个体索引数量必须与 G 的维度一致"))
    length(genotyped_indices) <= n || throw(ArgumentError("G 个体数不得超过系谱个体数"))

    A = numerator_relationship(ped)
    geno_idx = sort(collect(genotyped_indices))
    others = setdiff(collect(1:n), geno_idx)
    A22 = A[geno_idx, geno_idx]
    # 缩放 G 以匹配 A22
    g_trace = tr(G)
    a_trace = tr(A22)
    scale = g_trace ≈ 0 ? 1.0 : a_trace / g_trace
    Gt = tau * scale * G + omega * I

    A12 = isempty(others) ? zeros(0, length(geno_idx)) : A[others, geno_idx]
    A11 = isempty(others) ? zeros(0, 0) : A[others, others]

    inv_term = inv(Matrix(Gt)) - inv(Matrix(A22))
    top = isempty(others) ? Matrix(A22) : [A11 A12; transpose(A12) A22]
    adj = zeros(size(top))
    adj[end-length(geno_idx)+1:end, end-length(geno_idx)+1:end] = inv_term
    return Symmetric(top + adj)
end

# ------------------------------ 遗传评估核心 ------------------------------

"""
    mixed_model_solver(y, X, Z, R, G; use_cholesky::Bool = true, return_cuu::Bool = false)

求解混合模型方程：
```
|X'R⁻¹X    X'R⁻¹Z| |b| = |X'R⁻¹y|
|Z'R⁻¹X  Z'R⁻¹Z+G⁻¹| |u|   |Z'R⁻¹y|
```
返回固定效应 `b`、随机效应 `u`，若 `return_cuu=true`，额外返回随机效应方差矩阵 `Cuu`。
"""
function mixed_model_solver(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Z::AbstractMatrix{<:Real},
        R::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real}; use_cholesky::Bool = true,
        return_cuu::Bool = false)
    n = length(y)
    n == size(X, 1) == size(Z, 1) == size(R, 1) || throw(ArgumentError("观测数量与设计矩阵行数不一致"))

    Rinvy = R \ y
    RinvsX = R \ X
    RinvsZ = R \ Z

    XtRinvX = transpose(X) * RinvsX
    XtRinvZ = transpose(X) * RinvsZ
    ZtRinvZ = transpose(Z) * RinvsZ
    ZtRinvY = transpose(Z) * Rinvy
    XtRinvY = transpose(X) * Rinvy

    Ginv = inv(Matrix(G))
    lhs = [XtRinvX XtRinvZ; transpose(XtRinvZ) ZtRinvZ + Ginv]
    rhs = vcat(XtRinvY, ZtRinvY)

    if use_cholesky
        factor = cholesky(Symmetric(lhs))
        solution = factor \ rhs
        nb = size(X, 2)
        b = solution[1:nb]
        u = solution[nb+1:end]
        if return_cuu
            invlhs = inv(factor)
            Cuu = invlhs[nb+1:end, nb+1:end]
            return b, u, Symmetric(Cuu)
        end
        return b, u
    else
        solution = lhs \ rhs
        nb = size(X, 2)
        b = solution[1:nb]
        u = solution[nb+1:end]
        if return_cuu
            invlhs = inv(lhs)
            Cuu = invlhs[nb+1:end, nb+1:end]
            return b, u, Symmetric(Cuu)
        end
        return b, u
    end
end

"""
    reliability(G, Cuu)

根据预测遗传值方差 `Cuu` 与先验方差 `G` 计算个体育种值可靠度。
"""
function reliability(G::AbstractMatrix{<:Real}, Cuu::AbstractMatrix{<:Real})
    diagG = diag(G)
    diagC = diag(Cuu)
    all(diagG .> 0) || throw(ArgumentError("G 对角元素必须为正"))
    return clamp.(1 .- diagC ./ diagG, 0.0, 1.0)
end

"""
    kfold_crossvalidation(y, X, Z, R, G; k::Int = 5, rng = Random.default_rng())

执行 K 折交叉验证，返回平均预测准确度（皮尔逊相关）及标准差。
"""
function kfold_crossvalidation(y, X, Z, R, G; k::Int = 5, rng = Random.default_rng())
    n = length(y)
    k > 1 || throw(ArgumentError("K 折交叉验证的 K 必须大于 1"))
    k <= n || throw(ArgumentError("K 折交叉验证的 K 不得超过样本数量"))
    idx = collect(1:n)
    Random.shuffle!(rng, idx)
    fold_sizes = fill(div(n, k), k)
    for i in 1:rem(n, k)
        fold_sizes[i] += 1
    end
    folds = Vector{Vector{Int}}(undef, k)
    pos = 1
    for i in 1:k
        s = fold_sizes[i]
        folds[i] = idx[pos:pos + s - 1]
        pos += s
    end

    acc = Float64[]
    full_index = collect(1:n)
    for fold in folds
        train = setdiff(full_index, fold)
        y_train = y[train]
        X_train = X[train, :]
        Z_train = Z[train, :]
        R_train = R[train, train]
        Rinvy = R_train \ y_train
        RinvsX = R_train \ X_train
        RinvsZ = R_train \ Z_train

        XtRinvX = transpose(X_train) * RinvsX
        XtRinvZ = transpose(X_train) * RinvsZ
        ZtRinvZ = transpose(Z_train) * RinvsZ
        ZtRinvY = transpose(Z_train) * Rinvy
        XtRinvY = transpose(X_train) * Rinvy

        Ginv = inv(Matrix(G))
        lhs = [XtRinvX XtRinvZ; transpose(XtRinvZ) ZtRinvZ + Ginv]
        rhs = vcat(XtRinvY, ZtRinvY)

        sol = cholesky(Symmetric(lhs)) \ rhs
        nb = size(X, 2)
        b = sol[1:nb]
        u = sol[nb+1:end]

        y_pred = X[fold, :] * b + Z[fold, :] * u
        std_pred = std(y_pred)
        std_obs = std(y[fold])
        if std_pred == 0 || std_obs == 0
            push!(acc, 0.0)
        else
            push!(acc, cor(y_pred, y[fold]))
        end
    end
    return mean(acc), std(acc)
end

# ------------------------------ 品质控制与指标 ------------------------------

"""
    snp_quality_control(M; maf_threshold = 0.01, missing_rate = 0.05)

对基因型矩阵执行简单质量控制，返回过滤后的矩阵及保留的标记索引。
- `maf_threshold`：最小等位基因频率阈值；
- `missing_rate`：允许的最大缺失比例（以 `NaN` 表示缺失）。
"""
function snp_quality_control(M::AbstractMatrix{<:Real}; maf_threshold::Real = 0.01, missing_rate::Real = 0.05)
    n, m = size(M)
    kept_cols = Int[]
    kept_vectors = Vector{Vector{Float64}}()

    for j in 1:m
        missing_count = 0
        data = Vector{Float64}(undef, n)
        observed = Float64[]
        for i in 1:n
            val = float(M[i, j])
            if isnan(val)
                missing_count += 1
                data[i] = NaN
            else
                data[i] = val
                push!(observed, val)
            end
        end

        miss_rate = missing_count / n
        miss_rate <= missing_rate || continue
        isempty(observed) && continue

        maf = mean(observed) / 2
        maf = min(maf, 1 - maf)
        maf >= maf_threshold || continue

        mean_val = mean(observed)
        for i in 1:n
            if isnan(data[i])
                data[i] = mean_val
            end
        end

        push!(kept_cols, j)
        push!(kept_vectors, data)
    end

    result = Matrix{Float64}(undef, n, length(kept_vectors))
    for (idx, col) in enumerate(kept_vectors)
        result[:, idx] = col
    end
    return result, kept_cols
end

# ------------------------------ 近交系数与报告 ------------------------------

"""
    inbreeding_coefficients(ped::Pedigree)

通过分子关系矩阵的对角线元素计算个体近交系数 Fᵢ = Aᵢᵢ - 1。
"""
function inbreeding_coefficients(ped::Pedigree)
    A = numerator_relationship(ped)
    return diag(A) .- 1
end

"""
    summary(ped::Pedigree)

输出简要的族谱统计指标，帮助用户快速了解数据质量。
"""
function Base.summary(ped::Pedigree)
    n = length(ped.ids)
    founders = count(i -> ped.sire[i] == 0 && ped.dam[i] == 0, 1:n)
    max_depth = maximum(ped.depth)
    return "Pedigree with $n animals, $founders founders, maximum depth $max_depth"
end

export Pedigree, pedigree, numerator_relationship, inverse_numerator_relationship,
       genomic_relationship, hybrid_relationship, mixed_model_solver,
       reliability, kfold_crossvalidation, snp_quality_control,
       inbreeding_coefficients

end # module
