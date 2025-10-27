### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 1873c52e-fe63-11ee-325b-81d332a6f2b4
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, "..", ".."))
	using GenomicPrediction
	using PlutoUI
	using DataFrames
	using CSV
end

# ╔═╡ 866324b1-8b83-42e7-9d7e-07e590db2075
md"""
# 交互式基因组预测应用

欢迎使用 `GenomicPrediction.jl` 的交互式 Pluto 应用！

这个简单的应用演示了如何：
1.  上传基因型和表型数据。
2.  交互式地调整模型超参数。
3.  训练一个 GBLUP 模型。
4.  查看预测结果。
"""

# ╔═╡ 2e705b4a-a71d-4054-916c-17eb48c48a7b
md"### 1. 上传数据"

# ╔═╡ 489d81d6-d083-460d-a7f4-f5f284764b88
@bind geno_upload PlutoUI.FilePicker()

# ╔═╡ f9d57a9e-da45-4226-a079-c5d015951da6
@bind pheno_upload PlutoUI.FilePicker()

# ╔═╡ c3368297-b67e-402a-9694-523c0cd72635
md"""
**请上传您的数据：**

-   **基因型数据 (CSV):** $(geno_upload)
-   **表型数据 (CSV):** $(pheno_upload)
"""

# ╔═╡ a1917719-21d3-4673-a010-47b2c0199042
data = if geno_upload !== nothing && pheno_upload !== nothing
    try
        geno_df = CSV.read(geno_upload["data"], DataFrame)
        pheno_df = CSV.read(pheno_upload["data"], DataFrame)
        GenomicPrediction.GenomicData(geno_df, pheno_df)
    catch e
        Markdown.MD(Markdown.Admonition("error", "文件读取失败", [
            Markdown.p("请确保您上传的是有效的 CSV 文件，并且它们的格式正确。"),
            Markdown.Code(sprint(showerror, e))
        ]))
    end
else
    md"**等待文件上传...**"
end

# ╔═╡ 7c68884c-c1a7-47d0-863a-23b9ac3ed158
md"### 2. 设置模型参数"

# ╔═╡ b4d5f0b1-4217-49d6-953e-8c83a7f6f57e
@bind lambda Slider(1.0:1.0:200.0, default=10.0, show_value=true)

# ╔═╡ 6f2a831e-1f20-4e5a-a131-ab77095914fd
md"**GBLUP 正则化参数 (λ):** $(lambda)"

# ╔═╡ 6e3a479b-2224-4f01-9878-3a9d3e8e4526
md"### 3. 运行模型并查看结果"

# ╔═╡ 0f124b6e-44ab-46dd-a106-e7e025a74e54
results = if data isa GenomicPrediction.GenomicData
    md"""
    **点击下面的按钮来训练模型并生成预测。**
    $(@bind run_button Button("运行 GBLUP 分析"))
    """
else
    md"**请先成功上传数据。**"
end

# ╔═╡ 5d13f0c3-f421-4f81-831e-451996d92025
begin
    run_button
    if data isa GenomicPrediction.GenomicData
        model = GenomicPrediction.GBLUPModel(lambda)
        GenomicPrediction.fit!(model, data)
        predictions = GenomicPrediction.predict(model, data.genotypes)

        results_df = DataFrame(
            ID = data.genotypes.ID,
            TruePhenotype = data.phenotypes.y,
            PredictedPhenotype = predictions
        )

        corr_acc = cor(results_df.TruePhenotype, results_df.PredictedPhenotype)

        Markdown.MD(
            Markdown.H4("分析完成！"),
            Markdown.p("预测准确率 (相关系数): $(round(corr_acc, digits=4))"),
            Markdown.H4("预测结果预览:"),
            results_df
        )
    else
        " "
    end
end

# ╔═╡ Cell order:
# ╟─866324b1-8b83-42e7-9d7e-07e590db2075
# ╟─2e705b4a-a71d-4054-916c-17eb48c48a7b
# ╟─c3368297-b67e-402a-9694-523c0cd72635
# ╟─a1917719-21d3-4673-a010-47b2c0199042
# ╟─7c68884c-c1a7-47d0-863a-23b9ac3ed158
# ╟─6f2a831e-1f20-4e5a-a131-ab77095914fd
# ╟─b4d5f0b1-4217-49d6-953e-8c83a7f6f57e
# ╟─6e3a479b-2224-4f01-9878-3a9d3e8e4526
# ╟─0f124b6e-44ab-46dd-a106-e7e025a74e54
# ╟─5d13f0c3-f421-4f81-831e-451996d92025
# ╟─1873c52e-fe63-11ee-325b-81d332a6f2b4
# ╟─489d81d6-d083-460d-a7f4-f5f284764b88
# ╟─f9d57a9e-da45-4226-a079-c5d015951da6
