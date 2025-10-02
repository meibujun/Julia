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

# 流程
1.  根据模型定义构建设计矩阵 (X, Z) 和观测向量 (y)。
2.  如果遗传力(h²)或方差组分未知，则调用REML进行估计。
3.  根据方差组分构建并求解混合模型方程 (MME)。
4.  计算育种值的可靠性（当前为占位符）。
5.  将所有结果打包并返回。

# 参数
- `model::ModelSpec`: 模型规格。
- `dm::DataManager`: 数据管理器。
- `method::Symbol`: 评估方法，如 `:BLUP`, `:GBLUP`, `:SSGBLUP`。
- `h2::Union{Float64,Nothing}`: 用户提供的遗传力（如果已知）。
- `estimate_variances::Bool`: 是否强制使用REML估计方差。

# 返回
- `GeneticEvalResult`: 包含所有评估结果的对象。
"""
function run_evaluation(model::ModelSpec, dm::DataManager;
                       method::Symbol=:BLUP,
                       h2::Union{Float64,Nothing}=nothing,
                       estimate_variances::Bool=false)

    @info "="^70
    @info "开始遗传评估分析"
    @info "方法: $method"
    @info "="^70

    ensure_animal_map!(dm)

    # 1. 构建设计矩阵
    X, Z_dict, y = build_design_matrices(dm.phenotypes, model, dm.animal_map)

    # 2. 估计或设置方差组分
    variances = Dict{String,Float64}()
    if estimate_variances
        reml_result = estimate_variances_reml(X, Z_dict, y, dm, model)
        variances = reml_result.variance_components
    else
        h2_val = isnothing(h2) ? 0.3 : h2
        if isnothing(h2); @warn "未提供h2且未启用REML，使用默认值 h2=0.3"; end
        total_var = var(y)
        # 简化：假设只有一个加性效应
        variances["animal"] = total_var * h2_val
        variances["residual"] = total_var * (1 - h2_val)
    end

    # 3. 求解MME
    C, rhs = setup_mme(X, Z_dict, y, dm, model, variances)
    solutions = solve_mme(C, rhs)

    # 4. 提取结果和计算可靠性
    n_fixed = size(X, 2)
    fixed_effects = Dict("effects" => solutions[1:n_fixed])

    # 简化：假设只有一个随机效应
    u = solutions[n_fixed+1:end]
    # 使用DataManager中的全局动物列表来对齐ID
    animals_in_order = ordered_animals(dm)
    breeding_values_df = DataFrame(animal_id=animals_in_order, EBV=u)

    # 可靠性计算占位符
    reliability_df = DataFrame(animal_id=animals_in_order, reliability=zeros(length(u)))

    # 5. 返回结果对象
    result = GeneticEvalResult(
        breeding_values_df,
        fixed_effects,
        variances,
        reliability_df,
        Dict("converged"=>true), # 简化
        model,
        method
    )

    println(result) # 打印摘要
    return result
end

# 结果摘要的打印函数
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
    # 合并育种值和可靠性
    output_df = leftjoin(result.breeding_values, result.reliability, on=:animal_id)
    CSV.write(filepath, output_df)
    @info "结果已保存。"
end
