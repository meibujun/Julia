module Visualization

using Plots
using StatsPlots
using DataFrames
using Statistics

"""
    manhattan_plot(df; chr_col = :Chr, pos_col = :Pos, p_col = :P_value, title = "Manhattan Plot")

绘制稀有变异或上位性结果的曼哈顿图，自动标注基因组阈值并输出出版级图形。
"""
function manhattan_plot(df::DataFrame; chr_col = :Chr, pos_col = :Pos, p_col = :P_value, title = "Manhattan Plot")
    grouped = groupby(df, chr_col)
    xtick = Float64[]
    labels = String[]
    x = Float64[]
    y = Float64[]
    offset = 0.0
    for (chr, sub) in enumerate(grouped)
        positions = sub[:, pos_col] .+ offset
        append!(x, positions)
        append!(y, -log10.(sub[:, p_col] .+ eps()))
        push!(xtick, mean(positions))
        push!(labels, string(first(sub[!, chr_col])))
        offset = maximum(positions) + 1e5
    end
    p = scatter(x, y; color = :steelblue, ms = 4, legend = false, xlabel = "Chromosome", ylabel = "-log10(P)", title)
    hline!(p, [-log10(5e-8)], color = :red, linestyle = :dash)
    xticks!(p, xtick, labels)
    return p
end

"""
    qq_plot(pvalues)

生成 QQ 图用于评估统计检验的显著性分布。
"""
function qq_plot(pvalues::AbstractVector)
    expected = -log10.(collect(1:length(pvalues)) ./ (length(pvalues) + 1))
    observed = -log10.(sort(pvalues) .+ eps())
    p = scatter(expected, observed; color = :darkgreen, label = "Observed", xlabel = "Expected -log10(P)", ylabel = "Observed -log10(P)", title = "Q-Q Plot")
    plot!(p, expected, expected; color = :black, linestyle = :dash, label = "Ideal")
    return p
end

"""
    forest_plot(results; effect_col = :effect, se_col = :se, study_col = :study)

绘制森林图展示多研究效应量的一致性。
"""
function forest_plot(results::DataFrame; effect_col = :effect, se_col = :se, study_col = :study)
    effects = results[:, effect_col]
    ses = results[:, se_col]
    studies = string.(results[:, study_col])
    ci_low = effects .- 1.96 .* ses
    ci_high = effects .+ 1.96 .* ses
    p = plot(; xlabel = "Effect", ylabel = "Study", title = "Forest Plot", legend = false)
    for (i, study) in enumerate(studies)
        plot!(p, [ci_low[i], ci_high[i]], [i, i]; color = :navy)
        scatter!(p, [effects[i]], [i]; color = :orange, markerstrokecolor = :black)
    end
    ytick!(p, 1:length(studies), studies)
    return p
end

"""
    network_plot(edges; threshold = 1e-5)

基于上位性互作结果绘制网络图，突出显著互作关系。
"""
function network_plot(edges::DataFrame; threshold::Float64 = 1e-5)
    filtered = filter(row -> row[:P_value] <= threshold, edges)
    n = nrow(filtered)
    if n == 0
        return plot(title = "No significant interactions", legend = false)
    end
    θ = range(0, 2π; length = n + 1)[1:end-1]
    x = cos.(θ)
    y = sin.(θ)
    p = plot(; aspect_ratio = 1, legend = false, title = "Epistasis Network")
    for i in 1:n
        scatter!(p, [x[i]], [y[i]]; color = :skyblue, label = i == 1 ? "SNP" : "")
        annotate!(p, x[i], y[i], text(filtered[i, :SNP1] * "-" * filtered[i, :SNP2], 6, :black))
    end
    for i in 1:n
        for j in i+1:n
            plot!(p, [x[i], x[j]], [y[i], y[j]]; color = :gray, alpha = 0.3)
        end
    end
    return p
end

"""
    save_plot(path, plt)

将图形保存到指定路径，自动创建目录。
"""
function save_plot(path::AbstractString, plt)
    mkpath(dirname(path))
    Plots.savefig(plt, path)
    return path
end

end # module
