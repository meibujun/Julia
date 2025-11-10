# docs/make.jl

using Documenter
using GenomicPro

makedocs(
    sitename = "GenomicPro.jl",
    format = Documenter.HTML(),
    modules = [GenomicPro],
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ]
)
