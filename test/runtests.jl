using Test
using AnimalBreeding
using DataFrames
using DataFrames: Not
using Random
using CSV

@testset "AnimalBreeding" begin
    animals = ["A1", "A2", "A3", "A4"]
    phenotypes = DataFrame(animal = animals,
        herd = ["H1", "H1", "H2", "H2"],
        year = [2023, 2023, 2024, 2024],
        milk = [10.5, 11.2, 9.7, 10.9])
    pedigree = DataFrame(animal = animals,
        sire = [missing, "A1", "A1", "A2"],
        dam = [missing, missing, "A2", "A3"])
    genotypes = DataFrame(animal = animals,
        SNP1 = [0.0, 1.0, 2.0, 1.0],
        SNP2 = [1.0, 0.0, 1.0, 2.0])

    repo = DataRepository(phenotypes = phenotypes, pedigrees = pedigree, genotypes = genotypes)
    integrate_data!(repo)

    @testset "Data management" begin
        report = validate_data(repo)
        @test report[:phenotype_rows] == 4
        G, gids = compute_relationship_matrix(repo; type = :genomic, ids = animals)
        @test size(G) == (4, 4)
        A, aids = compute_relationship_matrix(repo; type = :pedigree, ids = animals)
        @test size(A) == (4, 4)
    end

    model = define_model(traits = [:milk], fixed = [:herd], random = [("animal", :additive)])

    @testset "Mixed model evaluation" begin
        res = run_evaluation(model, repo; method = :GBLUP, h2 = 0.35)
        @test nrow(res.breeding_values) == 4
        @test all(!isnan, res.breeding_values.reliability)
        res2 = run_evaluation(model, repo; method = :BLUP, h2 = 0.3)
        @test res2.trait == :milk
    end

    @testset "Bayesian sampling" begin
        bayes = run_bayesian_evaluation(model, repo; method = :BayesC, n_iter = 300, burn_in = 100, thin = 20)
        @test length(bayes.marker_effects) == 2
        stats = mcmc_diagnostics(bayes; parameter = :marker_variance)
        @test haskey(stats, :mean)
    end

    @testset "Machine learning" begin
        X = Matrix(select(repo.genotypes, Not(:animal)))
        y = Float64.(repo.phenotypes[:, :milk])
        rf = train_ml_model(:RandomForest, X, y; n_trees = 20, max_depth = 4)
        preds = predict(rf, X)
        @test length(preds) == length(y)
        cv_stats = cross_validate(X, y, :RandomForest; n_folds = 2, rng = MersenneTwister(42), n_trees = 20, max_depth = 4)
        @test haskey(cv_stats, :mean_correlation)
    end

    @testset "CLI" begin
        ph_file = tempname()
        ped_file = tempname()
        geno_file = tempname()
        CSV.write(ph_file, repo.phenotypes)
        CSV.write(ped_file, repo.pedigrees)
        CSV.write(geno_file, repo.genotypes)
        run_cli(["validate", "--phenotype", ph_file, "--pedigree", ped_file])
        run_cli(["evaluate", "--phenotype", ph_file, "--pedigree", ped_file, "--genotype", geno_file,
                 "--trait", "milk", "--method", "GBLUP", "--fixed", "herd", "--random", "animal", "--h2", "0.3"])
    end
end
