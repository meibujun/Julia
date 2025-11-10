# docs/make.jl

using Documenter
using GenomicPro

makedocs(
    sitename = "GenomicPro.jl",
    format = Documenter.HTML(),
    modules = [GenomicPro]
)
