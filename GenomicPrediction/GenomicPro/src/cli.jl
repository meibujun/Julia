# src/cli.jl

using ArgParse
using .GenomicPro

function main()
    s = ArgParseSettings()

    @add_arg_table s begin
        "gblup"
            help = "Run GBLUP analysis"
            action = :command
        "ssgblup"
            help = "Run SSGBLUP analysis"
            action = :command
    end

    @add_arg_table s["gblup"] begin
        "--geno"
            help = "Genotype file"
            required = true
        "--pheno"
            help = "Phenotype file"
            required = true
        "--trait"
            help = "Trait name"
            required = true
    end

    @add_arg_table s["ssgblup"] begin
        "--geno"
            help = "Genotype file"
            required = true
        "--pheno"
            help = "Phenotype file"
            required = true
        "--trait"
            help = "Trait name"
            required = true
        "--ped"
            help = "Pedigree file"
            required = true
    end

    args = parse_args(s)

    command = args["%COMMAND%"]
    command_args = args[command]

    if command == "gblup"
        geno = read_genotypes(command_args["geno"])
        pheno = read_phenotypes(command_args["pheno"], :ID, [Symbol(command_args["trait"])])

        G = compute_grm(geno)
        y = pheno.table[:, Symbol(command_args["trait"])]

        vc = estimate_variance_components(G, y)
        λ = vc.residual_variance / vc.genetic_variance

        results = solve_gblup(G, y, λ)
        println("Breeding values: ", results.breeding_values)

    elseif command == "ssgblup"
        geno = read_genotypes(command_args["geno"])
        pheno = read_phenotypes(command_args["pheno"], :ID, [Symbol(command_args["trait"])])
        ped = read_pedigree(command_args["ped"], :ID, :Sire, :Dam)

        # This is a simplified call, a real implementation would need to match IDs
        genotyped_indices = 1:size(geno, 1)

        G = compute_grm(geno)
        y = pheno.table[:, Symbol(command_args["trait"])]

        vc = estimate_variance_components(G, y[genotyped_indices])
        λ = vc.residual_variance / vc.genetic_variance

        results = solve_ssgblup(G, ped, genotyped_indices, y, λ)
        println("Breeding values: ", results.breeding_values)
    end
end

main()
