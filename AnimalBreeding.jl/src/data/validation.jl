# ============================================================================
# 数据模块 - 数据验证
# AnimalBreeding.jl
# ============================================================================

"""
    validate_data(dm::DataManager) -> Bool

对DataManager中所有已加载的数据进行全面的完整性和一致性检查。
"""
function validate_data(dm::DataManager)
    @info "开始全面的数据验证..."

    issues = String[]

    if isnothing(dm.phenotypes)
        push!(issues, "错误: 表型数据 (phenotypes) 未加载，无法进行评估。")
    end
    if isnothing(dm.pedigree) && isnothing(dm.genotypes)
        push!(issues, "错误: 必须至少加载谱系 (pedigree) 或基因型 (genotypes) 数据之一。")
    end

    if !isempty(issues)
        error("数据验证失败: \n" * join(issues, "\n"))
    end

    if !isnothing(dm.pedigree)
        update_animal_map_from_pedigree!(dm)
    elseif isempty(dm.animal_map)
        update_animal_map_from_pedigree!(dm)
    end

    all_animal_ids = Set(keys(dm.animal_map))
    @info "项目中共有 $(length(all_animal_ids)) 个独立个体。"

    pheno_animals = isnothing(dm.phenotypes) ? Set{Any}() : Set(dm.phenotypes.animal)

    if !isnothing(dm.pedigree)
        ped_animals = Set(dm.pedigree.animal)
        missing_in_ped = setdiff(pheno_animals, ped_animals)
        if !isempty(missing_in_ped)
            @warn "警告: $(length(missing_in_ped)) 个有表型记录的动物不在谱系文件中。"
        end
    end

    if !isnothing(dm.genotypes)
        geno_animals = Set(dm.genotypes[!, first(names(dm.genotypes))])
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
