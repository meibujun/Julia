using Documenter, OrthogonalGenomics

makedocs(
    sitename = "OrthogonalGenomics.jl",
    authors = "Mei Bujun",
    pages = [
        "Home" => "index.md",
        "Manual" => "manual.md",
        "API" => "api.md",
        "Examples" => "examples.md"
    ]
)

deploydocs(
    repo = "github.com/meibujun/OrthogonalGenomics.jl.git",
)
