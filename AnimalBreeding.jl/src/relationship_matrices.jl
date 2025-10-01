# --- 关系矩阵计算模块 ---
# 该模块负责计算遗传评估中所需的各种关系矩阵。
# 主要包括基于谱系的加性遗传关系矩阵 (A) 和基于基因组标记的基因组关系矩阵 (G)。

"""
    compute_relationship_matrix(dm::DataManager; type::Symbol, use_genotyped_only_for_g::Bool=true)

计算指定类型的关系矩阵，并将结果存储在 `DataManager` 对象中。
这是一个高级封装函数，根据用户请求调用相应的底层计算函数。

# 参数
- `dm::DataManager`: 存储所有数据的DataManager对象。
- `type::Symbol`: 指定要计算的矩阵类型。支持的类型包括:
    - `:pedigree` 或 `:A`: 计算谱系关系矩阵 A。
    - `:genomic` 或 `:G`: 计算基因组关系矩阵 G。
    - `:single_step` 或 `:H`: (占位) 计算单步关系矩阵 H (当前未实现)。
- `use_genotyped_only_for_g::Bool`: 计算G矩阵时，是否只使用在基因型文件中出现的个体。默认为`true`。

# 异常
- 如果请求的矩阵类型不受支持，或者计算所需的数据（如基因型数据）缺失，将抛出错误。
"""
function compute_relationship_matrix(dm::DataManager; type::Symbol, use_genotyped_only_for_g::Bool=true)
    println("--- 开始计算关系矩阵 ($type) ---")
    type_alias = Dict(:A => :pedigree, :G => :genomic, :H => :single_step)
    actual_type = get(type_alias, type, type)

    if actual_type == :pedigree
        if isempty(dm.pedigree)
            error("计算A矩阵需要谱系数据，但数据未加载。")
        end
        dm.A = calculate_a_matrix(dm.pedigree)
        println("成功计算 A 矩阵。")
    elseif actual_type == :genomic
        if isempty(dm.genotypes)
            error("计算G矩阵需要基因型数据，但数据未加载。")
        end
        dm.G = calculate_g_matrix(dm.genotypes, use_genotyped_only_for_g ? dm.genotypes.animal : dm.pedigree.animal)
        println("成功计算 G 矩阵。")
    elseif actual_type == :single_step
        # 此处为未来实现H矩阵的占位符
        error("单步法 (H矩阵) 的计算功能尚未实现。")
    else
        error("不支持的矩阵类型: $type。请从 `:pedigree` (`:A`), `:genomic` (`:G`) 中选择。")
    end
    println("--- 关系矩阵计算完成 ---")
end

"""
    calculate_a_matrix(pedigree::DataFrame) -> SparseMatrixCSC{Float64, Int}

根据谱系数据计算加性遗传关系矩阵 (A矩阵)。

此实现采用了高效的 **列表法 (tabular method)** 来构建A矩阵的逆矩阵 (A-inverse)，
因为在混合模型方程 (MME) 的求解中，直接使用的是A-inverse，这样可以避免对大型A矩阵进行求逆。
然而，为了教学和调试目的，本函数最后还是将A-inverse求逆得到了A矩阵。

**警告**: 对于大规模谱系 (如超过5000个体)，对A-inverse求逆会非常耗时且消耗大量内存。
在生产环境中，应使用直接处理稀疏A-inverse的迭代求解器。

# 参数
- `pedigree::DataFrame`: 包含 `animal`, `sire`, `dam` 列的谱系数据。

# 返回
- `SparseMatrixCSC{Float64, Int}`: 计算得到的A矩阵，以稀疏格式存储。
"""
function calculate_a_matrix(pedigree::DataFrame)
    # 创建谱系的副本以避免修改原始数据
    ped = copy(pedigree)
    # 将缺失的亲本ID统一表示为0，便于处理
    ped.sire = coalesce.(ped.sire, 0)
    ped.dam = coalesce.(ped.dam, 0)

    # 建立个体ID到整数索引的映射，这是构建矩阵的基础
    # 确保所有在谱系中出现的个体（包括作为亲本但无记录的个体）都有索引
    all_animals = sort(unique(vcat(ped.animal, ped.sire, ped.dam)))
    # 移除代表未知亲本的0
    filter!(x -> x != 0, all_animals)

    id_map = Dict(id => i for (i, id) in enumerate(all_animals))
    n = length(all_animals)

    # 构建A-inverse矩阵 (A⁻¹)
    # A⁻¹ 是一个稀疏矩阵，只有少数非零元素，因此使用稀疏格式存储效率很高
    A_inv = spzeros(n, n)

    # 遍历每个个体，根据Henderson法则填充A⁻¹矩阵
    for id in all_animals
        i = id_map[id]

        # 找到该个体的亲本
        sire_row = filter(row -> row.animal == id, ped)
        sire_id = isempty(sire_row) || ismissing(sire_row[1, :sire]) ? 0 : sire_row[1, :sire]
        dam_id = isempty(sire_row) || ismissing(sire_row[1, :dam]) ? 0 : sire_row[1, :dam]

        s = sire_id != 0 ? id_map[sire_id] : 0
        d = dam_id != 0 ? id_map[dam_id] : 0

        # 根据亲本情况应用不同的法则
        if s != 0 && d != 0 # 双亲已知
            A_inv[i, i] += 2.0
            A_inv[s, d] -= 0.5; A_inv[d, s] -= 0.5
            A_inv[i, s] += 1.0; A_inv[s, i] += 1.0
            A_inv[i, d] += 1.0; A_inv[d, i] += 1.0
        elseif s != 0 || d != 0 # 单亲已知
            p = max(s, d)
            A_inv[i, i] += 4/3
            A_inv[p, p] -= 1/3
            A_inv[i, p] += 1.0; A_inv[p, i] += 1.0
        else # 双亲未知 (基础动物)
            A_inv[i, i] += 1.0
        end
    end

    # Henderson法则构建的是L'DL，这里需要转换
    # L_inv = I - P, P是亲本指向矩阵
    # A_inv = L_inv' * D_inv * L_inv
    # 上述的构建方法是直接构建A_inv，但逻辑似乎有误，应使用更标准的算法
    # 修正为更直接的、基于贡献的构建方法

    # 重置A_inv
    A_inv = spzeros(n, n)
    d = zeros(n) # 对角线元素

    for i in 1:n
        animal_id = all_animals[i]
        sire_id, dam_id = 0, 0
        row = findfirst(ped.animal .== animal_id)
        if !isnothing(row)
            sire_id = coalesce(ped.sire[row], 0)
            dam_id = coalesce(ped.dam[row], 0)
        end

        s = sire_id != 0 ? id_map[sire_id] : 0
        d = dam_id != 0 ? id_map[dam_id] : 0

        if s != 0 && d != 0
            d[i] = 0.5 - 0.25 * (d[s] + d[d])
        elseif s != 0 || d != 0
            p = max(s, d)
            d[i] = 0.75 - 0.25 * d[p]
        else
            d[i] = 1.0
        end

        val = 1.0 / d[i]
        A_inv[i, i] = val
        if s != 0; A_inv[i, s] = A_inv[s, i] = -0.5 * val; end
        if d != 0; A_inv[i, d] = A_inv[d, i] = -0.5 * val; end
        if s != 0 && d != 0; A_inv[s, d] = A_inv[d, s] = A_inv[s, d] + 0.25 * val; end
    end


    if n > 5000
        @warn "谱系规模过大 ($n x $n)。直接对A-inverse求逆会非常缓慢且消耗大量内存。建议在分析中直接使用A-inverse。"
    end

    # 将稀疏的A-inverse转换为稠密矩阵后求逆，得到A
    try
        # LU分解求解是比直接inv()更稳定和高效的方式
        A = sparse(inv(Matrix(A_inv)))
        return A
    catch e
        error("对 A-inverse 求逆失败。谱系可能存在问题（如导致矩阵奇异）。错误: $e")
    end
end


"""
    calculate_g_matrix(genotypes::DataFrame, animal_list::Vector) -> Matrix{Float64}

根据基因型数据计算基因组关系矩阵 (G矩阵)，采用 VanRaden (2008) 的方法一。

该方法是基因组选择中的标准方法。计算公式为:
G = Z * Z' / (2 * Σ(pᵢ * (1 - pᵢ)))
其中:
- `Z` 是中心化的基因型矩阵 (M - P)。
- `M` 是原始基因型矩阵 (个体 x 标记)。
- `P` 是基于等位基因频率计算的期望基因型矩阵。
- `pᵢ` 是第 i 个标记的等位基因频率。

# 参数
- `genotypes::DataFrame`: 包含个体ID和SNP标记的DataFrame。
- `animal_list::Vector`: 一个向量，指定哪些动物需要包含在G矩阵中，并决定其顺序。

# 返回
- `Matrix{Float64}`: 计算得到的G矩阵，是一个稠密的方阵。
"""
function calculate_g_matrix(genotypes::DataFrame, animal_list::Vector)
    # 筛选出在指定列表中的动物
    geno_subset = filter(row -> row.animal in animal_list, genotypes)
    if isempty(geno_subset)
        error("基因型文件中没有找到任何一个指定的动物。")
    end

    # 保证G矩阵的顺序与 animal_list 一致
    order_map = Dict(id => i for (i, id) in enumerate(animal_list))
    sort!(geno_subset, :animal, by = x -> order_map[x])

    # 提取标记矩阵 (假定第一列是动物ID)
    M = Matrix(geno_subset[:, 2:end])
    n_animals, n_markers = size(M)

    # 计算每个标记的等位基因频率 (p)
    # 假设编码为0, 1, 2, 等位基因为B, p是B的频率
    # p = (2 * count(2) + 1 * count(1)) / (2 * n_animals)
    p = sum(M, dims=1) ./ (2 * n_animals)

    # 处理单态标记 (频率为0或1)，这些标记对计算关系没有贡献，且会导致分母为0
    non_mono_indices = findall(f -> f > 1e-6 && f < 1 - 1e-6, vec(p))
    if length(non_mono_indices) < n_markers
        @warn "$(n_markers - length(non_mono_indices)) 个单态标记被检测到，将从G矩阵计算中移除。"
        M = M[:, non_mono_indices]
        p = p[:, non_mono_indices]
        n_markers = length(non_mono_indices)
        if n_markers == 0
            error("所有标记都是单态的，无法计算G矩阵。")
        end
    end

    # 创建期望基因型矩阵 P
    # P 的每一行都是 2p'
    P = 2 .* p

    # 中心化标记矩阵 M -> Z
    Z = M .- P

    # 计算分母: 2 * Σ(pᵢ * (1 - pᵢ))
    denominator = 2 * sum(p .* (1 .- p))

    # 计算 G 矩阵
    G = (Z * Z') / denominator

    return G
end