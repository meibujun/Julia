### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ Cell ID
md"""
# GenomicPrediction.jl Pluto 教程
通过交互式控件探索基因组预测流程。
"""

# ╔═╡ Cell ID
begin
    using GenomicPrediction
    using Random
    using PlutoUI
    using DataFrames
end

# ╔═╡ Cell ID
@bind seed Slider(1:100, default=42)

# ╔═╡ Cell ID
@bind heritability Slider(0.1:0.1:0.9, default=0.5)

# ╔═╡ Cell ID
md"""
当前随机种子: **$(seed)**, 遗传力: **$(heritability)**
"""

# ╔═╡ Cell ID
begin
    Random.seed!(seed)
    dataset = simulate_genomic_data(150, 300; h2 = heritability)
    dataset
end

# ╔═╡ Cell ID
begin
    model = GBLUPModel(λ = 0.7)
    fit!(model, dataset.genotype, dataset.phenotype)
    preds = predict(model, dataset.genotype)
    metrics = evaluate_metrics(dataset.phenotype, preds)
    md"预测指标: ``$(metrics)``"
end

# ╔═╡ Cell ID
begin
    pipeline = default_workflow(dataset)
    run_autogs(pipeline, dataset)
    DataFrame(pipeline.results)
end
