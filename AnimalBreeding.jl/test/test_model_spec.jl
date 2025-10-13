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

@testset "ModelSpec 构造函数兼容性" begin
    fixed = ["herd", "year"]
    random = [RandomEffect("animal", :additive)]

    legacy = ModelSpec(traits=["milk"], fixed_effects=fixed, random=random)
    modern = ModelSpec(traits=["milk"], fixed=fixed, random_effects=random)

    @test legacy.fixed_effects == fixed
    @test legacy.random_effects == random
    @test modern.fixed_effects == fixed
    @test modern.random_effects == random
end
