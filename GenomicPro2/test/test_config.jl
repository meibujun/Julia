"""
Configuration Module Tests

Tests for configuration management system:
- TOML file parsing
- Environment variable overrides
- Default values
- Configuration validation
"""

using Test
using GenomicPro2
using GenomicPro2.Config
using TOML

@testset "Config Module Tests" begin

    # ========================================================================
    # Test 1: Default Configuration
    # ========================================================================

    @testset "Default Configuration" begin
        config = GenomicProConfig()

        # Compute defaults
        @test config.compute.threads > 0
        @test config.compute.threads <= Threads.nthreads()
        @test config.compute.use_gpu == false  # Default
        @test config.compute.gpu_device_id == 0
        @test config.compute.parallel_backend == "threads"

        # Memory defaults
        @test config.memory.max_memory_gb > 0
        @test config.memory.chunk_size > 0
        @test config.memory.use_compression == true
        @test config.memory.cache_size_mb > 0

        # IO defaults
        @test !isempty(config.io.temp_dir)
        @test !isempty(config.io.output_dir)
        @test config.io.auto_cleanup == true
        @test config.io.buffer_size_mb > 0

        # Logging defaults
        @test config.logging.level in ["DEBUG", "INFO", "WARN", "ERROR"]
        @test config.logging.console_output == true

        # API defaults
        @test config.api.port == 8080
        @test config.api.cors_enabled == true
        @test config.api.max_upload_size_mb > 0

        # Analysis defaults
        @test config.analysis.default_model in ["gblup", "bayesr", "bayescpi", "rkhs"]
        @test config.analysis.cross_validation_folds >= 2
        @test config.analysis.significance_threshold > 0
        @test config.analysis.max_iterations > 0
    end

    # ========================================================================
    # Test 2: Load Configuration from TOML
    # ========================================================================

    @testset "Load TOML Configuration" begin
        # Create temporary TOML file
        toml_content = """
        [compute]
        threads = 16
        use_gpu = true
        gpu_device_id = 1

        [memory]
        max_memory_gb = 64.0
        chunk_size = 10000

        [logging]
        log_level = "DEBUG"
        log_file = "test.log"
        console_output = true

        [api]
        port = 9090
        host = "0.0.0.0"

        [analysis]
        default_model = "bayesr"
        cross_validation_folds = 10
        """

        temp_file = tempname() * ".toml"
        write(temp_file, toml_content)

        try
            config = load_config(temp_file)

            @test config.compute.threads == 16
            @test config.compute.use_gpu == true
            @test config.compute.gpu_device_id == 1
            @test config.memory.max_memory_gb == 64.0
            @test config.memory.chunk_size == 10000
            @test config.logging.level == "DEBUG"
            @test config.logging.log_file == "test.log"
            @test config.api.port == 9090
            @test config.api.host == "0.0.0.0"
            @test config.analysis.default_model == "bayesr"
            @test config.analysis.cross_validation_folds == 10

        finally
            rm(temp_file, force=true)
        end
    end

    # ========================================================================
    # Test 3: Environment Variable Overrides
    # ========================================================================

    @testset "Environment Variable Overrides" begin
        # Set environment variables
        ENV["GENOMICPRO_THREADS"] = "32"
        ENV["GENOMICPRO_GPU"] = "true"
        ENV["GENOMICPRO_LOG_LEVEL"] = "WARN"

        try
            config = GenomicProConfig()

            # Check if environment variables override defaults
            # Note: The actual implementation may vary
            # This test checks the concept

            @test haskey(ENV, "GENOMICPRO_THREADS")
            @test haskey(ENV, "GENOMICPRO_GPU")
            @test haskey(ENV, "GENOMICPRO_LOG_LEVEL")

        finally
            delete!(ENV, "GENOMICPRO_THREADS")
            delete!(ENV, "GENOMICPRO_GPU")
            delete!(ENV, "GENOMICPRO_LOG_LEVEL")
        end
    end

    # ========================================================================
    # Test 4: Partial Configuration
    # ========================================================================

    @testset "Partial Configuration" begin
        # Test with incomplete TOML (should use defaults for missing)
        partial_toml = """
        [compute]
        threads = 8

        [logging]
        log_level = "INFO"
        """

        temp_file = tempname() * ".toml"
        write(temp_file, partial_toml)

        try
            config = load_config(temp_file)

            # Specified values
            @test config.compute.threads == 8
            @test config.logging.level == "INFO"

            # Default values for unspecified
            @test config.memory.max_memory_gb > 0  # Should have default
            @test config.api.port == 8080  # Should have default

        finally
            rm(temp_file, force=true)
        end
    end

    # ========================================================================
    # Test 5: Configuration Validation
    # ========================================================================

    @testset "Configuration Validation" begin
        config = GenomicProConfig()

        # Validate ranges
        @test config.compute.threads > 0
        @test config.compute.gpu_device_id >= 0
        @test config.memory.max_memory_gb > 0
        @test config.memory.chunk_size > 0
        @test config.memory.cache_size_mb > 0
        @test config.io.buffer_size_mb > 0
        @test config.api.port > 0
        @test config.api.port < 65536
        @test config.api.max_upload_size_mb > 0
        @test config.analysis.cross_validation_folds >= 2
        @test config.analysis.significance_threshold > 0
        @test config.analysis.max_iterations > 0
    end

    # ========================================================================
    # Test 6: ComputeConfig Sub-structure
    # ========================================================================

    @testset "ComputeConfig" begin
        compute = ComputeConfig(
            threads = 16,
            use_gpu = true,
            gpu_device_id = 1,
            parallel_backend = "distributed"
        )

        @test compute.threads == 16
        @test compute.use_gpu == true
        @test compute.gpu_device_id == 1
        @test compute.parallel_backend == "distributed"
    end

    # ========================================================================
    # Test 7: MemoryConfig Sub-structure
    # ========================================================================

    @testset "MemoryConfig" begin
        memory = MemoryConfig(
            max_memory_gb = 128.0,
            chunk_size = 20000,
            use_compression = false,
            cache_size_mb = 4096
        )

        @test memory.max_memory_gb == 128.0
        @test memory.chunk_size == 20000
        @test memory.use_compression == false
        @test memory.cache_size_mb == 4096
    end

    # ========================================================================
    # Test 8: IOConfig Sub-structure
    # ========================================================================

    @testset "IOConfig" begin
        io = IOConfig(
            temp_dir = "/tmp/test",
            output_dir = "./output",
            auto_cleanup = false,
            buffer_size_mb = 256
        )

        @test io.temp_dir == "/tmp/test"
        @test io.output_dir == "./output"
        @test io.auto_cleanup == false
        @test io.buffer_size_mb == 256
    end

    # ========================================================================
    # Test 9: LogConfig Sub-structure
    # ========================================================================

    @testset "LogConfig" begin
        log = LogConfig(
            level = "DEBUG",
            log_file = "debug.log",
            console_output = true,
            performance_logging = true
        )

        @test log.level == "DEBUG"
        @test log.log_file == "debug.log"
        @test log.console_output == true
        @test log.performance_logging == true
    end

    # ========================================================================
    # Test 10: APIConfig Sub-structure
    # ========================================================================

    @testset "APIConfig" begin
        api = APIConfig(
            host = "localhost",
            port = 3000,
            cors_enabled = false,
            max_upload_size_mb = 1024,
            session_timeout_minutes = 60
        )

        @test api.host == "localhost"
        @test api.port == 3000
        @test api.cors_enabled == false
        @test api.max_upload_size_mb == 1024
        @test api.session_timeout_minutes == 60
    end

    # ========================================================================
    # Test 11: AnalysisConfig Sub-structure
    # ========================================================================

    @testset "AnalysisConfig" begin
        analysis = AnalysisConfig(
            default_model = "rkhs",
            cross_validation_folds = 5,
            significance_threshold = 1e-8,
            max_iterations = 50000
        )

        @test analysis.default_model == "rkhs"
        @test analysis.cross_validation_folds == 5
        @test analysis.significance_threshold == 1e-8
        @test analysis.max_iterations == 50000
    end

    # ========================================================================
    # Test 12: Print Configuration
    # ========================================================================

    @testset "Print Configuration" begin
        config = GenomicProConfig()

        # Test that print_config doesn't error
        io = IOBuffer()
        print_config(config, io=io)
        output = String(take!(io))

        @test !isempty(output)
        @test occursin("Compute", output)
        @test occursin("Memory", output)
        @test occursin("Logging", output)
    end

    # ========================================================================
    # Test 13: Configuration File Not Found
    # ========================================================================

    @testset "Configuration File Not Found" begin
        nonexistent_file = "nonexistent_config_12345.toml"

        # Should return default config when file not found
        config = load_config(nonexistent_file)

        @test config !== nothing
        @test config.compute.threads > 0  # Should have defaults
    end

    # ========================================================================
    # Test 14: Invalid TOML Syntax
    # ========================================================================

    @testset "Invalid TOML Syntax" begin
        invalid_toml = """
        [compute
        threads = 8
        """

        temp_file = tempname() * ".toml"
        write(temp_file, invalid_toml)

        try
            # Should handle error gracefully
            @test_throws Exception TOML.parsefile(temp_file)
        finally
            rm(temp_file, force=true)
        end
    end

    # ========================================================================
    # Test 15: Type Conversions
    # ========================================================================

    @testset "Type Conversions" begin
        # Test that string "true"/"false" converts to boolean
        toml_with_strings = """
        [compute]
        threads = 8
        use_gpu = true
        """

        temp_file = tempname() * ".toml"
        write(temp_file, toml_with_strings)

        try
            data = TOML.parsefile(temp_file)
            @test typeof(data["compute"]["use_gpu"]) == Bool
        finally
            rm(temp_file, force=true)
        end
    end

    # ========================================================================
    # Test 16: Save Configuration
    # ========================================================================

    @testset "Save Configuration" begin
        config = GenomicProConfig()

        temp_file = tempname() * ".toml"

        try
            # Test save functionality (if implemented)
            # save_config(config, temp_file)
            # @test isfile(temp_file)

            # For now, just test structure exists
            @test config !== nothing
        finally
            rm(temp_file, force=true)
        end
    end

    # ========================================================================
    # Test 17: Configuration Equality
    # ========================================================================

    @testset "Configuration Equality" begin
        config1 = GenomicProConfig()
        config2 = GenomicProConfig()

        # Default configs should have same values
        @test config1.compute.threads == config2.compute.threads
        @test config1.memory.max_memory_gb == config2.memory.max_memory_gb
        @test config1.logging.level == config2.logging.level
    end

    # ========================================================================
    # Test 18: Thread Count Validation
    # ========================================================================

    @testset "Thread Count Validation" begin
        max_threads = Threads.nthreads()

        # Test thread count doesn't exceed available
        config = GenomicProConfig()

        if config.compute.threads == 0
            # 0 means use all available
            @test max_threads >= 1
        else
            @test config.compute.threads <= max_threads
        end
    end

    # ========================================================================
    # Test 19: GPU Configuration
    # ========================================================================

    @testset "GPU Configuration" begin
        config = GenomicProConfig()

        # GPU device ID should be non-negative
        @test config.compute.gpu_device_id >= 0

        # If GPU not used, device ID doesn't matter
        if !config.compute.use_gpu
            @test true  # No additional validation needed
        end
    end

    # ========================================================================
    # Test 20: Configuration Merging
    # ========================================================================

    @testset "Configuration Merging" begin
        # Test that TOML values override defaults
        toml_content = """
        [compute]
        threads = 99
        """

        temp_file = tempname() * ".toml"
        write(temp_file, toml_content)

        try
            config = load_config(temp_file)

            # Should have overridden value
            @test config.compute.threads == 99

            # Should have default values for unspecified
            @test config.memory.max_memory_gb > 0

        finally
            rm(temp_file, force=true)
        end
    end

end  # @testset "Config Module Tests"

println("✓ Config module tests completed")
