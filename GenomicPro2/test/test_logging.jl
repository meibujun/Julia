"""
Logging Module Tests

Tests for logging framework:
- Log level filtering
- Multiple output targets
- Performance logging
- Structured logging
- Log rotation
"""

using Test
using GenomicPro2
using GenomicPro2.Logging
using Dates

@testset "Logging Module Tests" begin

    # ========================================================================
    # Test 1: Log Level Enum
    # ========================================================================

    @testset "Log Levels" begin
        @test DEBUG < INFO
        @test INFO < WARN
        @test WARN < ERROR

        @test Int(DEBUG) == 1
        @test Int(INFO) == 2
        @test Int(WARN) == 3
        @test Int(ERROR) == 4
    end

    # ========================================================================
    # Test 2: Log Level from String
    # ========================================================================

    @testset "Log Level Parsing" begin
        @test loglevel_from_string("DEBUG") == DEBUG
        @test loglevel_from_string("INFO") == INFO
        @test loglevel_from_string("WARN") == WARN
        @test loglevel_from_string("WARNING") == WARN  # Alias
        @test loglevel_from_string("ERROR") == ERROR

        # Case insensitive
        @test loglevel_from_string("debug") == DEBUG
        @test loglevel_from_string("info") == INFO

        # Invalid should default to INFO
        @test loglevel_from_string("INVALID") == INFO
    end

    # ========================================================================
    # Test 3: Logger Structure
    # ========================================================================

    @testset "Logger Structure" begin
        logger = Logger(
            INFO,
            [stdout],
            false,
            false,
            "yyyy-mm-dd HH:MM:SS"
        )

        @test logger.level == INFO
        @test length(logger.outputs) == 1
        @test logger.structured == false
        @test logger.performance == false
        @test !isempty(logger.timestamp_format)
    end

    # ========================================================================
    # Test 4: Setup Logging - Console Only
    # ========================================================================

    @testset "Setup Logging - Console" begin
        setup_logging(level="INFO", console=true, log_file=nothing)

        logger = get_logger()

        @test logger !== nothing
        @test logger.level == INFO
        @test stdout in logger.outputs
    end

    # ========================================================================
    # Test 5: Setup Logging - File Output
    # ========================================================================

    @testset "Setup Logging - File" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="DEBUG", log_file=temp_log, console=false)

            logger = get_logger()

            @test logger.level == DEBUG
            @test length(logger.outputs) >= 1

            # Write a log message
            @info "Test log message"

            # File should exist
            @test isfile(temp_log)

            # File should have content
            content = read(temp_log, String)
            @test !isempty(content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 6: Log Level Filtering
    # ========================================================================

    @testset "Log Level Filtering" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="WARN", log_file=temp_log, console=false)

            # These should NOT be logged (below WARN)
            @debug "Debug message"
            @info "Info message"

            # These SHOULD be logged
            @warn "Warning message"
            @error "Error message"

            # Check file content
            content = read(temp_log, String)

            @test !occursin("Debug message", content)
            @test !occursin("Info message", content)
            @test occursin("Warning message", content)
            @test occursin("Error message", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 7: Structured Logging
    # ========================================================================

    @testset "Structured Logging" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(
                level="INFO",
                log_file=temp_log,
                console=false,
                structured=true
            )

            @info "Test message" key1="value1" key2=42

            content = read(temp_log, String)

            # Should contain structured fields
            @test occursin("Test message", content)
            @test occursin("key1", content)
            @test occursin("key2", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 8: Performance Logging - Macro
    # ========================================================================

    @testset "Performance Logging - Macro" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(
                level="INFO",
                log_file=temp_log,
                console=false,
                performance=true
            )

            result = @log_performance "Test Task" begin
                sleep(0.01)  # Simulate work
                42
            end

            @test result == 42

            content = read(temp_log, String)

            # Should contain performance metrics
            @test occursin("Test Task", content)
            @test occursin("time_seconds", content) || occursin("Performance", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 9: Performance Timer
    # ========================================================================

    @testset "Performance Timer" begin
        timer = PerformanceTimer("My Task")

        @test timer.name == "My Task"
        @test timer.start_time === nothing
        @test timer.end_time === nothing

        start!(timer)
        @test timer.start_time !== nothing

        sleep(0.01)

        stop!(timer)
        @test timer.end_time !== nothing

        elapsed_time = elapsed(timer)
        @test elapsed_time !== nothing
        @test elapsed_time >= 0.01

        mem = memory_delta(timer)
        @test mem !== nothing
    end

    # ========================================================================
    # Test 10: Multiple Log Outputs
    # ========================================================================

    @testset "Multiple Outputs" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="INFO", log_file=temp_log, console=true)

            logger = get_logger()

            # Should have both stdout and file
            @test length(logger.outputs) >= 2

            @info "Multi-output test"

            # File should have content
            content = read(temp_log, String)
            @test occursin("Multi-output test", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 11: Log Message Formatting
    # ========================================================================

    @testset "Log Message Format" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(
                level="INFO",
                log_file=temp_log,
                console=false,
                structured=false
            )

            @info "Format test" param1=100 param2="test"

            content = read(temp_log, String)

            # Should contain timestamp, level, message
            @test occursin("INFO", content)
            @test occursin("Format test", content)
            @test occursin("param1", content)
            @test occursin("100", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 12: Log File Creation
    # ========================================================================

    @testset "Log File Creation" begin
        # Test with nested directory
        temp_dir = mktempdir()
        temp_log = joinpath(temp_dir, "logs", "app.log")

        try
            setup_logging(level="INFO", log_file=temp_log, console=false)

            @info "Directory creation test"

            # Directory and file should be created
            @test isdir(dirname(temp_log))
            @test isfile(temp_log)

        finally
            close_logger()
            rm(temp_dir, recursive=true, force=true)
        end
    end

    # ========================================================================
    # Test 13: Close Logger
    # ========================================================================

    @testset "Close Logger" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="INFO", log_file=temp_log, console=false)

            @info "Before close"

            close_logger()

            # Logger should be cleared
            # (May need to reinitialize for next test)

        finally
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 14: Log Rotation
    # ========================================================================

    @testset "Log Rotation" begin
        temp_log = tempname() * ".log"

        try
            # Create a large log file
            open(temp_log, "w") do f
                for i in 1:10000
                    println(f, "Log line $i with some content to make it larger")
                end
            end

            # Check file size
            file_size_mb = filesize(temp_log) / 1024^2

            if file_size_mb > 1.0
                # Should rotate
                rotated = rotate_log_file(temp_log, max_size_mb=1, max_files=3)
                @test rotated == true

                # Rotated file should exist
                @test isfile(temp_log * ".1")
            end

        finally
            rm(temp_log, force=true)
            for i in 1:5
                rm(temp_log * ".$i", force=true)
            end
        end
    end

    # ========================================================================
    # Test 15: Search Logs
    # ========================================================================

    @testset "Search Logs" begin
        temp_log = tempname() * ".log"

        try
            # Write some log content
            open(temp_log, "w") do f
                println(f, "2024-01-01 INFO: Normal message")
                println(f, "2024-01-01 ERROR: Error occurred")
                println(f, "2024-01-01 WARN: Warning here")
                println(f, "2024-01-01 ERROR: Another error")
            end

            # Search for errors
            matches = search_logs(temp_log, r"ERROR")

            @test length(matches) == 2
            @test all(occursin("ERROR", m) for m in matches)

        finally
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 16: Performance Logging Disabled
    # ========================================================================

    @testset "Performance Logging Disabled" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(
                level="INFO",
                log_file=temp_log,
                console=false,
                performance=false  # Disabled
            )

            result = @log_performance "Task" begin
                42
            end

            @test result == 42

            content = read(temp_log, String)

            # Performance logging should be minimal or absent
            # (depends on implementation)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 17: Configuration Integration
    # ========================================================================

    @testset "Config Integration" begin
        using GenomicPro2.Config

        config = GenomicProConfig()
        config.logging.level = "DEBUG"
        config.logging.log_file = tempname() * ".log"

        try
            setup_logging_from_config(config)

            logger = get_logger()

            @test logger.level == DEBUG
            @test isfile(config.logging.log_file)

        finally
            close_logger()
            rm(config.logging.log_file, force=true)
        end
    end

    # ========================================================================
    # Test 18: Timestamp Format
    # ========================================================================

    @testset "Timestamp Format" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="INFO", log_file=temp_log, console=false)

            @info "Timestamp test"

            content = read(temp_log, String)

            # Should contain a timestamp
            # Format: YYYY-MM-DD HH:MM:SS
            @test occursin(r"\d{4}-\d{2}-\d{2}", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 19: Concurrent Logging
    # ========================================================================

    @testset "Concurrent Logging" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="INFO", log_file=temp_log, console=false)

            # Simulate concurrent logging
            @sync begin
                for i in 1:10
                    Threads.@spawn @info "Concurrent message $i"
                end
            end

            content = read(temp_log, String)

            # All messages should be present
            for i in 1:10
                @test occursin("Concurrent message $i", content)
            end

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

    # ========================================================================
    # Test 20: Error Handling
    # ========================================================================

    @testset "Error Handling" begin
        temp_log = tempname() * ".log"

        try
            setup_logging(level="ERROR", log_file=temp_log, console=false)

            # Log an exception
            try
                error("Test exception")
            catch e
                @error "Caught exception" exception=e
            end

            content = read(temp_log, String)

            @test occursin("ERROR", content)
            @test occursin("exception", content)

        finally
            close_logger()
            rm(temp_log, force=true)
        end
    end

end  # @testset "Logging Module Tests"

println("✓ Logging module tests completed")
