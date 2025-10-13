using Test
using AnimalBreeding

@testset "模型规格构建" begin
    model = define_model(traits=["milk"],
                         fixed=["herd", "year"],
                         random=[("animal", :additive)])
    @test isa(model, ModelSpec)
    @test length(model.random_effects) == 1
    @test model.random_effects[1].name == "animal"
    @test model.random_effects[1].type == :additive

    extra = RandomEffect("herd_year", :iid)
    model2 = define_model(traits=["growth"],
                          fixed=["batch"],
                          random=[extra])
    @test model2.random_effects[1] === extra

    @test_throws ErrorException define_model(random=[(:invalid,)])
end
