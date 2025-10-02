# ============================================================================
# BLUP评估模块 - 主评估函数
# AnimalBreeding.jl
# ============================================================================

"""
    GeneticEvalResult

存储BLUP/GBLUP评估结果的结构体。
"""
struct GeneticEvalResult
    breeding_values::DataFrame
    fixed_effects::Dict{String,Vector{Float64}}
    variances::Dict{String,Float64}
    reliability::DataFrame
    convergence::Dict{String,Any}
    model::ModelSpec
    method::Symbol
end

"""
    run_evaluation(model::ModelSpec, dm::DataManager; ...) -> GeneticEvalResult

运行遗传评估分析的主函数。
"""
function run_evaluation(model::ModelSpec, dm::DataManager;
                       method::Symbol=:BLUP,
                       h2::Union{Float64,Nothing}=nothing,
                       estimate_variances::Bool=false)

    @info "="^70
    @info "开始遗传评估分析"
    @info "方法: $method"
    @info "="^70

    if isempty(dm.animal_map)
        update_animal_map_from_pedigree!(dm)
    end

    X, Z_dict, y = build_design_matrices(dm.phenotypes, model, dm.animal_map)

    variances = Dict{String,Float64}()
    if estimate_variances
        reml_result = estimate_variances_reml(X, Z_dict, y, dm, model)
        variances = reml_result.variance_components
    else
        h2_val = isnothing(h2) ? 0.3 : h2
        if isnothing(h2)
            @warn "未提供h2且未启用REML，使用默认值 h2=0.3"
        end
        total_var = var(y)
        for effect in model.random_effects
            if effect.type == :additive
                variances[effect.name] = total_var * h2_val
            else
                variances[effect.name] = total_var * 0.1
            end
        end
        variances["residual"] = total_var * (1 - h2_val)
    end

    C, rhs = setup_mme(X, Z_dict, y, dm, model, variances)
    solutions = solve_mme(C, rhs)

    n_fixed = size(X, 2)
    fixed_effects = Dict("effects" => solutions[1:n_fixed])

    random_effect_values = Dict{String, Vector{Float64}}()
    current_pos = n_fixed
    for effect in model.random_effects
        dim = size(Z_dict[effect.name], 2)
        random_effect_values[effect.name] = solutions[current_pos+1:current_pos+dim]
        current_pos += dim
    end

    if isempty(random_effect_values)
        breeding_values_df = DataFrame(animal_id=Vector{Any}(), EBV=Float64[])
        reliability_df = DataFrame(animal_id=Vector{Any}(), reliability=Float64[])
    else
        ordered_animals = Vector{Any}(undef, length(dm.animal_map))
        for (id, idx) in dm.animal_map
            ordered_animals[idx] = id
        end

        if haskey(random_effect_values, "animal")
            u = random_effect_values["animal"]
        else
            first_key = first(keys(random_effect_values))
            u = random_effect_values[first_key]
        end

        breeding_values_df = DataFrame(animal_id=ordered_animals, EBV=u)
        reliability_df = DataFrame(animal_id=ordered_animals, reliability=zeros(length(u)))
    end

    result = GeneticEvalResult(
        breeding_values_df,
        fixed_effects,
        variances,
        reliability_df,
        Dict("converged"=>true),
        model,
        method,
    )

    println(result)
    return result
end

function Base.show(io::IO, result::GeneticEvalResult)
    println(io, "\n--- 遗传评估结果摘要 ---")
    println(io, "方法: ", result.method)
    println(io, "性状: ", join(result.model.traits, ", "))
    if haskey(result.variances, "animal") && haskey(result.variances, "residual")
        h2 = result.variances["animal"] / (result.variances["animal"] + result.variances["residual"])
        println(io, "估计的遗传力: ", round(h2, digits=3))
    end
    println(io, "育种值 (EBV) 均值: ", round(mean(result.breeding_values.EBV), digits=4))
    println(io, "可靠性均值: ", round(mean(result.reliability.reliability), digits=4))
    println(io, "------------------------")
end

"""
    save_results(result::GeneticEvalResult, filepath::String)

将评估结果保存到文件。
"""
function save_results(result::GeneticEvalResult, filepath::String)
    @info "保存结果到: $filepath"
    output_df = leftjoin(result.breeding_values, result.reliability, on=:animal_id)
    CSV.write(filepath, output_df)
    @info "结果已保存。"
end
