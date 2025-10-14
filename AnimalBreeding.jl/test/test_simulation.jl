using Test
using AnimalBreeding
using Statistics

@testset "模拟数据生成" begin
    dm, params = simulate_complete_dataset(n_generations=3, n_per_generation=40, n_markers=120, n_qtl=8, h2=0.35, add_omics=true)

    @test isa(dm, DataManager)
    @test !isnothing(dm.pedigree)
    @test !isnothing(dm.genotypes)
    @test !isnothing(dm.phenotypes)
    @test nrow(dm.pedigree) == length(dm.pedigree.animal)
    @test nrow(dm.phenotypes) == nrow(dm.pedigree)
    @test size(dm.genotypes, 1) == nrow(dm.pedigree)
    @test size(dm.genotypes, 2) == 1 + 120 # 包含 animal_id 列

    # 检查真实参数结构
    @test haskey(params, "phenotype")
    @test haskey(params["phenotype"], "true_breeding_values")
    @test length(params["phenotype"]["true_breeding_values"]) == nrow(dm.phenotypes)

    # 验证表型与TBV之间的相关性符合设定遗传力方向
    tbv = params["phenotype"]["true_breeding_values"]
    trait = dm.phenotypes.trait
    @test !isapprox(var(tbv), 0.0)
    @test cor(tbv, trait) > 0.5

    # 检查 animal_map 已填充
    @test !isempty(dm.animal_map)
    @test sort(collect(keys(dm.animal_map))) == collect(1:nrow(dm.pedigree))

    # 若生成多组学数据，验证其维度
    if haskey(dm.omics_data, "transcriptome")
        omics_df = dm.omics_data["transcriptome"]
        @test nrow(omics_df) == nrow(dm.pedigree)
        @test first(names(omics_df)) == :animal
    end
end
