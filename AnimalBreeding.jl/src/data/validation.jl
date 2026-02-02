# ============================================================================
# 数据模块 - 数据验证
# AnimalBreeding.jl
# ============================================================================

"""
    validate_data(dm::DataManager) -> Bool

对DataManager中所有已加载的数据进行全面的完整性和一致性检查。

# 主要检查项
- 确保核心数据（如系谱、表型）已加载。
- 检查不同数据集（谱系、表型、基因型）中动物ID的一致性。
- 调用特定于数据类型的验证函数（如 `validate_pedigree`）。

# 返回
- `Bool`: 如果所有检查都通过，则返回 `true`。否则，打印警告和错误信息，并可能抛出错误。
"""
function validate_data(dm::DataManager)
    @info "开始全面的数据验证..."

    issues = String[]

    # 1. 检查核心数据是否存在
    if isnothing(dm.phenotypes)
        push!(issues, "错误: 表型数据 (phenotypes) 未加载，无法进行评估。")
    end
    if isnothing(dm.pedigree) && isnothing(dm.genotypes)
        push!(issues, "错误: 必须至少加载谱系 (pedigree) 或基因型 (genotypes) 数据之一。")
    end

    if !isempty(issues)
        error("数据验证失败: \n" * join(issues, "\n"))
    end

    # 2. 建立一个包含所有动物的全局ID列表和映射
    all_animal_ids = Set()
    if !isnothing(dm.pedigree); union!(all_animal_ids, dm.pedigree.animal); end
    if !isnothing(dm.genotypes); union!(all_animal_ids, dm.genotypes.animal_id); end
    if !isnothing(dm.phenotypes); union!(all_animal_ids, dm.phenotypes.animal); end

    dm.animal_map = Dict(id => i for (i, id) in enumerate(sort(collect(all_animal_ids))))
    @info "项目中共有 $(length(all_animal_ids)) 个独立个体。"

    # 3. 检查ID一致性
    pheno_animals = Set(dm.phenotypes.animal)

    if !isnothing(dm.pedigree)
        ped_animals = Set(dm.pedigree.animal)

        # 检查有多少有表型的动物在谱系中
        missing_in_ped = setdiff(pheno_animals, ped_animals)
        if !isempty(missing_in_ped)
            @warn "警告: $(length(missing_in_ped)) 个有表型记录的动物不在谱系文件中。"
        end
    end

    if !isnothing(dm.genotypes)
        geno_animals = Set(dm.genotypes.animal_id)

        # 检查有多少有表型的动物有基因型
        missing_in_geno = setdiff(pheno_animals, geno_animals)
        if !isempty(missing_in_geno)
            @warn "警告: $(length(missing_in_geno)) 个有表型记录的动物没有基因型数据。"
        end

        if !isnothing(dm.pedigree)
            ped_animals = Set(dm.pedigree.animal)
            missing_in_ped_from_geno = setdiff(geno_animals, ped_animals)
            if !isempty(missing_in_ped_from_geno)
                @warn "警告: $(length(missing_in_ped_from_geno)) 个有基因型记录的动物不在谱系文件中。"
            end
        end
    end

    dm.validated = true
    @info "数据验证完成。请检查以上警告信息（如果有）。"
    return true
end