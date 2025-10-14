# ============================================================================
# 测试: REML 方差组分估计
# ============================================================================

using Test
using AnimalBreeding
using DataFrames

@testset "REML 方差估计" begin
    ped_df = DataFrame(
        animal = [1, 2, 3, 4],
        sire   = [0, 0, 1, 1],
        dam    = [0, 0, 2, 2],
    )

    phenotypes = DataFrame(
        animal = [1, 2, 3, 4],
        herd   = ["A", "A", "B", "B"],
        milk   = [1000.0, 980.0, 1120.0, 1090.0],
    )

    dm = DataManager()
    dm.pedigree = ped_df
    dm.phenotypes = phenotypes

    compute_relationship_matrix(dm, type=:pedigree, compute_inverse=true)

    model = define_model(
        traits = ["milk"],
        fixed = ["herd"],
        random = [("animal", :additive)],
    )

    X, Z_dict, y = build_design_matrices(dm.phenotypes, model, dm.animal_map)

    result = estimate_variances_reml(X, Z_dict, y, dm, model; max_iter=50, tol=1e-5)

    @test result.converged == true
    @test haskey(result.variance_components, "animal")
    @test result.variance_components["animal"] > 0
    @test result.variance_components["residual"] > 0
end
