"""
Tests for Core module.
"""

using GenomicPro2
using Test

function test_core()
    @testset "GenotypeValue" begin
        # Valid values
        @test GenotypeValue(0).value == 0x00
        @test GenotypeValue(1).value == 0x01
        @test GenotypeValue(2).value == 0x02
        @test ismissing(GenotypeValue(missing).value)

        # Invalid value
        @test_throws DomainError GenotypeValue(3)
        @test_throws DomainError GenotypeValue(-1)

        # Equality
        @test GenotypeValue(1) == GenotypeValue(1)
        @test GenotypeValue(0) != GenotypeValue(2)
    end

    @testset "AlleleFrequency" begin
        # Valid frequencies
        @test AlleleFrequency(0.0).value == 0.0
        @test AlleleFrequency(0.5).value == 0.5
        @test AlleleFrequency(1.0).value == 1.0

        # Invalid frequencies
        @test_throws DomainError AlleleFrequency(-0.1)
        @test_throws DomainError AlleleFrequency(1.1)

        # is_rare function
        @test is_rare(AlleleFrequency(0.005))  # < 0.01
        @test is_rare(AlleleFrequency(0.995))  # > 0.99
        @test !is_rare(AlleleFrequency(0.3))   # common

        # Custom threshold
        @test is_rare(AlleleFrequency(0.04), 0.05)
    end

    @testset "ValidationResult" begin
        result = ValidationResult()

        @test result.valid == true
        @test isempty(result.errors)
        @test isempty(result.warnings)

        # Add error
        add_error!(result, "Test error")
        @test result.valid == false
        @test length(result.errors) == 1

        # Add warning (doesn't fail validation)
        result2 = ValidationResult()
        add_warning!(result2, "Test warning")
        @test result2.valid == true
        @test length(result2.warnings) == 1
    end

    @testset "Validation merge" begin
        r1 = ValidationResult()
        add_error!(r1, "Error 1")

        r2 = ValidationResult()
        add_warning!(r2, "Warning 1")

        merged = merge([r1, r2])

        @test merged.valid == false  # One has error
        @test length(merged.errors) == 1
        @test length(merged.warnings) == 1
    end

    @testset "Exceptions" begin
        # DataValidationError
        err = DataValidationError("Test message", :field_name, 42)
        @test err.msg == "Test message"
        @test err.field == :field_name
        @test err.value == 42

        # DimensionMismatchError
        err = DimensionMismatchError((100, 1000), (100, 900))
        @test err.expected == (100, 1000)
        @test err.actual == (100, 900)

        # ConvergenceError
        err = ConvergenceError("Did not converge", 1000, 1e-3)
        @test err.iterations == 1000
        @test err.residual == 1e-3
    end
end
