using Documenter
using GenomicPrediction

DocMeta.setdocmeta!(GenomicPrediction, :DocTestSetup, :(using GenomicPrediction); recursive=true)

makedocs(
    sitename = "GenomicPrediction.jl 文档",
    authors = "GenomicPrediction.jl Contributors",
    format = Documenter.HTML(
        prettyurls = false,
        canonical = "https://example.com/GenomicPrediction.jl"
    ),
    modules = [GenomicPrediction],
    pages = [
        "简介" => "index.md",
        "快速入门" => "tutorial.md",
        "API 参考" => "api.md"
    ]
)

