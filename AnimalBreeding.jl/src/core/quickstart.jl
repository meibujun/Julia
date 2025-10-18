# =============================================================================
# 快速入门管线
# =============================================================================

"""
    quickstart_gblup(; kwargs...) -> (GeneticEvalResult, DataManager, Dict)

运行 README 中的快速入门流程：模拟数据、计算关系矩阵、定义模型并执行
GBLUP 评估。函数返回评估结果、填充后的 `DataManager` 与用于验证的真实
参数字典。如果提供 `save_path`，则会将育种值与可靠性保存为 CSV 文件。
"""
function quickstart_gblup(; n_generations::Int=5,
                          n_per_generation::Int=200,
                          n_markers::Int=5000,
                          h2::Float64=0.3,
                          trait_name::AbstractString="trait",
                          save_path::Union{Nothing,AbstractString}=nothing)
    dm, true_params = simulate_complete_dataset(
        n_generations=n_generations,
        n_per_generation=n_per_generation,
        n_markers=n_markers,
        h2=h2,
        trait_name=trait_name,
    )

    compute_relationship_matrix(dm, type=:genomic)

    model = define_model(
        traits=[trait_name],
        fixed=["herd"],
        random=[("animal", :additive)],
    )

    result = run_evaluation(model, dm, method=:GBLUP, h2=h2)

    if !isnothing(save_path)
        save_results(result, String(save_path))
    end

    return result, dm, true_params
end
