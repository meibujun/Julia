#!/usr/bin/env julia
#=!
    使用 Dash.jl 构建的简易 Web 界面, 展示预测结果与指标。
=#

using Dash
using DashCoreComponents
using DashHtmlComponents
using PlotlyJS
using GenomicPrediction
using Random

Random.seed!(2026)
dataset = simulate_genomic_data(120, 200; h2 = 0.6)
model = GBLUPModel(λ = 0.8)
fit!(model, dataset.genotype, dataset.phenotype)
preds = predict(model, dataset.genotype)
metrics = evaluate_metrics(dataset.phenotype, preds)

app = dash()

app.layout = html_div() do
    html_h1("GenomicPrediction.jl Dash Demo"),
    dcc_markdown("模型指标: ``$(metrics)``"),
    dcc_graph(
        id = "prediction-scatter",
        figure = PlotlyJS.plot(
            scatter(x = dataset.phenotype, y = preds, mode = "markers", name = "预测 vs 真值"),
            Layout(title = "预测对比", xaxis_title = "真实表型", yaxis_title = "预测表型")
        )
    )
end

run_server(app, "0.0.0.0", debug = true)
