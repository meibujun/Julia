# =============================================================================
# 数据模块 - 模拟数据生成
# AnimalBreeding.jl
# =============================================================================

"""
    simulate_complete_dataset(; kwargs...) -> (DataManager, Dict)

生成一个包含谱系、基因型、表型（可选多组学）数据的完整模拟数据集，
用于快速测试和演示 AnimalBreeding.jl 的工作流程。返回值为 `DataManager`
实例及一个存储真实参数的字典。

# 关键参数
- `n_generations::Int=3`: 模拟的世代数，至少为1。
- `n_per_generation::Int=80`: 每个世代的基础个体数。
- `n_animals::Union{Nothing,Int}=nothing`: 若指定则优先生效，用于直接控制总样本量。
- `n_markers::Int=500`: SNP 标记数量，至少为20个。
- `n_qtl::Int=12`: 用于生成真实育种值 (TBV) 的数量性状位点数，自动截断不超过标记数。
- `h2::Float64=0.3`: 目标遗传力 (0到1)，用于控制残差方差规模。
- `add_omics::Bool=true`: 是否生成一个简单的转录组数据表并附加到 `omics_data`。

# 返回
- `(dm, true_params)`: `dm` 为填充后的 `DataManager`，`true_params` 为包含
  真值（如 TBV、环境效应等）的字典，便于评估预测准确度。
"""
function simulate_complete_dataset(; n_generations::Int=3,
                                    n_per_generation::Int=80,
                                    n_animals::Union{Nothing,Int}=nothing,
                                    n_markers::Int=500,
                                    n_qtl::Int=12,
                                    h2::Float64=0.3,
                                    add_omics::Bool=true)
    n_generations < 1 && throw(ArgumentError("n_generations 至少为1"))
    n_per_generation < 1 && throw(ArgumentError("n_per_generation 至少为1"))
    n_markers < 20 && throw(ArgumentError("n_markers 至少为20"))
    h2 < 0 && throw(ArgumentError("h2 不能为负数"))
    n_qtl = clamp(n_qtl, 1, n_markers)
    requested_generations = n_generations

    # ---------------------------------------------------------------------
    # 确定各世代样本数
    # ---------------------------------------------------------------------
    if isnothing(n_animals)
        generation_sizes = fill(n_per_generation, n_generations)
        n_total = sum(generation_sizes)
    else
        n_animals < 1 && throw(ArgumentError("n_animals 必须大于0"))
        n_total = n_animals
        n_generations = min(n_generations, n_animals)
        base = fld(n_animals, n_generations)
        generation_sizes = fill(base, n_generations)
        remainder = n_animals - sum(generation_sizes)
        for g in 1:remainder
            generation_sizes[g] += 1
        end
    end

    # ---------------------------------------------------------------------
    # 构建谱系信息
    # ---------------------------------------------------------------------
    animal_ids = collect(1:n_total)
    generation_of = zeros(Int, n_total)
    rows = Vector{NamedTuple{(:animal, :sire, :dam),Tuple{Int,Int,Int}}}()

    idx = 1
    prev_generation = Int[]
    for g in 1:n_generations
        n_g = generation_sizes[g]
        curr_ids = animal_ids[idx:idx + n_g - 1]
        for (local_idx, animal) in enumerate(curr_ids)
            generation_of[animal] = g
            if g == 1 || isempty(prev_generation)
                push!(rows, (animal=animal, sire=0, dam=0))
            else
                sire_idx = ((local_idx - 1) % length(prev_generation)) + 1
                dam_idx = ((local_idx - 1 + ceil(Int, length(prev_generation) / 2)) % length(prev_generation)) + 1
                sire = prev_generation[sire_idx]
                dam = prev_generation[dam_idx]
                if sire == dam
                    dam = prev_generation[(dam_idx % length(prev_generation)) + 1]
                end
                push!(rows, (animal=animal, sire=sire, dam=dam))
            end
        end
        prev_generation = curr_ids
        idx += n_g
    end

    pedigree_df = DataFrame(rows)

    # ---------------------------------------------------------------------
    # 生成确定性的基因型矩阵 (0/1/2 编码)
    # ---------------------------------------------------------------------
    genotype_matrix = Matrix{Float64}(undef, n_total, n_markers)
    for (i, animal) in enumerate(animal_ids)
        for marker in 1:n_markers
            base_val = (animal * 11 + marker * 7 + generation_of[animal] * 3) % 3
            genotype_matrix[i, marker] = Float64(base_val)
        end
    end

    genotype_df = DataFrame(; animal_id = animal_ids)
    for marker in 1:n_markers
        genotype_df[!, Symbol("marker_" * string(marker))] = genotype_matrix[:, marker]
    end

    # ---------------------------------------------------------------------
    # 构建真实的遗传值 (TBV) 与环境效应
    # ---------------------------------------------------------------------
    qtl_indices = collect(1:n_qtl)
    additive_effects = [0.4 + 0.05 * cos(0.7 * j) for j in 1:n_qtl]
    centered_genotypes = genotype_matrix[:, qtl_indices] .- 1.0
    true_breeding_values = centered_genotypes * additive_effects

    genetic_var = var(true_breeding_values)
    if genetic_var < eps()
        genetic_var = 1.0
        true_breeding_values .= range(-0.5, 0.5; length=n_total)
    end

    residual_var = if h2 <= 0
        max(genetic_var, 1.0)
    elseif h2 >= 1
        eps()
    else
        genetic_var * (1 - h2) / h2
    end
    env_raw = [sin((animal_ids[i] + 3 * generation_of[animal_ids[i]]) * 0.37) for i in 1:n_total]
    env_raw .-= mean(env_raw)
    env_scale = ifelse(var(env_raw) > 0, sqrt(residual_var / var(env_raw)), 0.0)
    environmental_effects = env_raw .* env_scale

    # ---------------------------------------------------------------------
    # 固定效应 (如 herd) 与最终表型
    # ---------------------------------------------------------------------
    n_herds = clamp(round(Int, sqrt(n_total / 10)) + 1, 2, max(2, min(8, n_total)))
    herd_labels = ["H" * lpad(string(h), 2, '0') for h in 1:n_herds]
    herd_effect_levels = [0.25 * cos(π * (h - 1) / max(1, n_herds - 1)) for h in 1:n_herds]
    herd_assignment = similar(animal_ids, String)
    herd_effects = similar(true_breeding_values)
    for (i, animal) in enumerate(animal_ids)
        herd_idx = ((i - 1) % n_herds) + 1
        herd_assignment[i] = herd_labels[herd_idx]
        herd_effects[i] = herd_effect_levels[herd_idx]
    end

    trait_values = true_breeding_values .+ herd_effects .+ environmental_effects

    phenotypes_df = DataFrame(
        animal = animal_ids,
        herd = herd_assignment,
        trait = trait_values,
        TBV = true_breeding_values,
        herd_effect = herd_effects,
        residual = environmental_effects,
    )

    # ---------------------------------------------------------------------
    # 构造 DataManager 并填充元数据
    # ---------------------------------------------------------------------
    dm = DataManager()
    dm.pedigree = pedigree_df
    dm.genotypes = genotype_df
    dm.phenotypes = phenotypes_df
    dm.metadata["simulation"] = Dict(
        "requested_generations" => requested_generations,
        "effective_generations" => n_generations,
        "generation_sizes" => generation_sizes,
        "n_markers" => n_markers,
        "n_qtl" => n_qtl,
        "heritability" => h2,
    )

    true_params = Dict(
        "pedigree" => Dict(
            "generation_of" => generation_of,
        ),
        "genotype" => Dict(
            "qtl_indices" => qtl_indices,
            "qtl_effects" => additive_effects,
        ),
        "phenotype" => Dict(
            "true_breeding_values" => copy(true_breeding_values),
            "herd_effects" => copy(herd_effects),
            "residual_effects" => copy(environmental_effects),
            "trait" => copy(trait_values),
        ),
    )

    if add_omics
        n_features = min(10, max(4, Int(floor(n_markers / 50))))
        omics_df = DataFrame(; animal = animal_ids)
        for feature in 1:n_features
            base_col = genotype_matrix[:, feature]
            omics_df[!, Symbol("gene_" * string(feature))] = 0.3 .* base_col .+ 0.1 .* true_breeding_values .+
                0.05 .* cos.((feature .* base_col .+ animal_ids) .* 0.15)
        end
        dm.omics_data["transcriptome"] = omics_df
    end

    update_animal_map_from_pedigree!(dm)
    dm.validated = true
    dm.metadata["true_params"] = true_params

    return dm, true_params
end
