# =============================================================================
# 数据模拟模块 - 为示例与测试生成一致的数据结构
# =============================================================================

"""
    simulate_complete_dataset(; kwargs...) -> (DataManager, Dict{String,Any})

生成一个包含谱系、基因型、表型以及可选多组学数据的完整 `DataManager`
实例。该函数为文档、示例和测试提供稳定的数据来源，同时返回用于
验证分析结果的真实参数。

# 关键参数
- `n_generations::Int=3`: 谱系的代数。
- `n_per_generation::Int=100`: 每代动物数量。
- `n_animals::Union{Int,Nothing}=nothing`: 若提供，则覆盖总动物数并在给定代数
  内平均分配。
- `n_markers::Int=500`: 模拟基因型标记数量。
- `n_qtl::Int=min(n_markers, 25)`: 设定为定量性状位点的标记数量。
- `h2::Float64=0.3`: 目标表型遗传力。
- `add_omics::Bool=true`: 是否生成示例多组学数据。
- `trait_name::AbstractString="trait"`: 表型列名称。
- `seed::Union{Int,Nothing}=nothing`: 随机种子，便于重现。

返回 `(dm, true_params)`，其中 `dm` 为填充完成的 `DataManager`，
`true_params` 是包含真实遗传参数和模拟设置的字典。
"""
function simulate_complete_dataset(; n_generations::Int=3,
                                    n_per_generation::Int=100,
                                    n_animals::Union{Int,Nothing}=nothing,
                                    n_markers::Int=500,
                                    n_qtl::Int=min(n_markers, 25),
                                    h2::Float64=0.3,
                                    add_omics::Bool=true,
                                    trait_name::AbstractString="trait",
                                    seed::Union{Int,Nothing}=nothing)
    n_generations < 1 && error("n_generations 至少为 1。")
    n_markers < 1 && error("n_markers 至少为 1。")

    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    total_animals = isnothing(n_animals) ? n_generations * n_per_generation : n_animals
    total_animals < n_generations && error("总动物数必须不少于代数。")

    # 根据代数分配每代动物数量，保证总数匹配
    gen_sizes = fill(max(1, fld(total_animals, n_generations)), n_generations)
    remainder = total_animals - sum(gen_sizes)
    for i in 1:remainder
        gen_sizes[i] += 1
    end

    animal_ids = Vector{Int}()
    sires = Vector{Int}()
    dams = Vector{Int}()
    generation_animals = Vector{Vector{Int}}(undef, n_generations)

    current_id = 1
    for gen in 1:n_generations
        generation_animals[gen] = Int[]
        for _ in 1:gen_sizes[gen]
            sire = 0
            dam = 0
            if gen > 1 && !isempty(generation_animals[gen - 1])
                previous = generation_animals[gen - 1]
                sire = rand(rng, previous)
                dam = rand(rng, previous)
                if sire == dam && length(previous) > 1
                    candidates = filter(!=(sire), previous)
                    dam = rand(rng, candidates)
                end
            end
            push!(animal_ids, current_id)
            push!(sires, sire)
            push!(dams, dam)
            push!(generation_animals[gen], current_id)
            current_id += 1
        end
    end

    pedigree = DataFrame((
        animal = animal_ids,
        sire = sires,
        dam = dams,
    ))

    # 生成基因型矩阵
    marker_names = ["marker_$(lpad(string(i), 4, '0'))" for i in 1:n_markers]
    allele_freqs = 0.05 .+ 0.4 .* rand(rng, n_markers)
    geno_matrix = Array{Float64}(undef, total_animals, n_markers)
    for j in 1:n_markers
        p = allele_freqs[j]
        for i in 1:total_animals
            geno_matrix[i, j] = (rand(rng) < p ? 1 : 0) + (rand(rng) < p ? 1 : 0)
        end
    end

    genotype_df = DataFrame(animal_id = animal_ids)
    for (name, col) in zip(marker_names, eachcol(geno_matrix))
        genotype_df[!, name] = col
    end

    # 选择QTL并生成真实遗传效应
    n_qtl = clamp(n_qtl, 1, n_markers)
    qtl_indices = sort(randperm(rng, n_markers)[1:n_qtl])
    marker_effects = zeros(Float64, n_markers)
    marker_effects[qtl_indices] .= randn(rng, n_qtl)

    # 居中基因型后计算真实育种值
    centered_genotypes = copy(geno_matrix)
    for j in 1:n_markers
        centered_genotypes[:, j] .-= 2 * allele_freqs[j]
    end
    true_breeding_values = centered_genotypes * marker_effects
    bv_var = var(true_breeding_values)
    if bv_var < eps()
        true_breeding_values .+= randn(rng, total_animals) .* 1e-3
        bv_var = var(true_breeding_values)
    end

    # 调整效应规模以匹配目标遗传力
    if bv_var > 0
        scale = sqrt(h2 / bv_var)
        marker_effects .*= scale
        true_breeding_values = centered_genotypes * marker_effects
        bv_var = var(true_breeding_values)
    end

    # 模拟群体(herd)效应
    n_herds = clamp(div(total_animals, 50), 1, max(1, min(total_animals, 5)))
    herd_ids = repeat(1:n_herds, inner=ceil(Int, total_animals / n_herds))[1:total_animals]
    shuffle!(rng, herd_ids)
    herd_effects = randn(rng, n_herds) .* sqrt(bv_var * 0.2)
    herd_component = herd_effects[herd_ids]

    residual_var = max(bv_var * (1 - h2) / max(h2, eps()), 1e-6)
    phenotypes = true_breeding_values + herd_component + randn(rng, total_animals) .* sqrt(residual_var)

    phenotype_df = DataFrame((
        animal = animal_ids,
        herd = herd_ids,
        TBV = true_breeding_values,
    ))
    phenotype_df[!, Symbol(trait_name)] = phenotypes

    dm = DataManager()
    dm.pedigree = pedigree
    dm.genotypes = genotype_df
    dm.phenotypes = phenotype_df
    dm.metadata["allele_frequencies"] = allele_freqs
    dm.metadata["marker_names"] = marker_names
    dm.metadata["simulation"] = Dict(
        "n_generations" => n_generations,
        "gen_sizes" => gen_sizes,
        "h2" => h2,
        "seed" => seed,
    )
    dm.validated = true
    dm.animal_map = Dict(animal_ids .=> collect(1:total_animals))

    if add_omics
        transcriptome = DataFrame(animal = animal_ids)
        for idx in 1:10
            transcriptome[!, Symbol("gene_$(lpad(string(idx), 3, '0'))")] =
                true_breeding_values .* (0.3 + 0.1 * rand(rng)) + randn(rng, total_animals)
        end
        dm.omics_data["transcriptome"] = transcriptome
    end

    true_params = Dict(
        "markers" => Dict(
            "effects" => marker_effects,
            "qtl_indices" => qtl_indices,
            "allele_frequencies" => allele_freqs,
        ),
        "phenotype" => Dict(
            "true_breeding_values" => true_breeding_values,
            "residual_variance" => residual_var,
        ),
        "herd" => Dict(
            "assignments" => herd_ids,
            "effects" => herd_effects,
        ),
    )

    return dm, true_params
end
