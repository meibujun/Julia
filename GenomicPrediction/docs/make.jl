# docs/make.jl
using Documenter
using GenomicPrediction

# 设置文档生成参数
makedocs(
    sitename = "GenomicPrediction.jl",
    modules = [GenomicPrediction],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    pages = [
        "主页" => "index.md",
        "快速入门教程" => "tutorial.md",
        "API 参考" => "api.md",
    ]
)

# 可选：自动部署文档到 GitHub Pages
# deploydocs(
#     repo = "github.com/your-username/GenomicPrediction.jl.git",
# )
