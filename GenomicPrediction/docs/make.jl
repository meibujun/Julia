# docs/make.jl
using Documenter
using GenomicPrediction

makedocs(
    sitename = "GenomicPrediction.jl",
    format = Documenter.HTML(),
    modules = [GenomicPrediction],
    pages = [
        "主页" => "index.md",
        "API 参考" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/YourUser/GenomicPrediction.jl.git",
)
