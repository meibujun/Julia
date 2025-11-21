#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GenomicPro

function print_usage()
    println("Usage: genomicpro_cli.jl <pipeline_config.json>")
end

function main()
    isempty(ARGS) && return (print_usage(); exit(1))
    config_path = ARGS[1]
    config = read_pipeline_config(config_path)
    result = run_pipeline(config)
    println(render_summary(result))
end

main()
