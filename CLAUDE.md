# CLAUDE.md - AI Assistant Guide for Julia Repository

**Last Updated:** 2025-11-15
**Repository:** meibujun/Julia

## Repository Overview

This is a Julia programming language repository currently in its early stages. The repository contains minimal files and is set up for future Julia development.

### Current State

- **Primary Language:** Julia
- **Current Files:** readme.md
- **Project Status:** Early stage / Starter repository
- **Main Branch:** Not specified (default branch)
- **Development Branch:** claude/claude-md-mi0bulaexf0g5nv4-015mGhm324Krpiqwg4a5dnSD

## Repository Structure

### Expected Julia Project Structure

As this project develops, it should follow standard Julia package conventions:

```
Julia/
├── src/                    # Source code
│   └── Julia.jl           # Main module file
├── test/                   # Test files
│   └── runtests.jl        # Main test file
├── docs/                   # Documentation
│   ├── make.jl            # Documentation build script
│   └── src/               # Documentation source
├── examples/              # Example scripts
├── Project.toml           # Package dependencies and metadata
├── Manifest.toml          # Exact dependency versions (git-ignored for libraries)
├── README.md              # Project overview
├── LICENSE                # License file
└── .gitignore            # Git ignore patterns
```

## Julia Development Conventions

### 1. Package Management

**Project.toml**
- Defines package metadata, dependencies, and compatibility
- Required for any Julia package
- Should include name, uuid, version, and dependencies

**Manifest.toml**
- Auto-generated file with exact dependency versions
- Should be committed for applications
- Should be git-ignored for libraries/packages

### 2. Module Structure

Julia code should be organized in modules:

```julia
module YourModuleName

# Exports
export function_name, TypeName

# Imports
using PackageName
import AnotherPackage: specific_function

# Include source files
include("submodule.jl")
include("utils.jl")

# Module code here

end # module
```

### 3. Code Style Guidelines

Follow Julia community conventions:

- **Naming Conventions:**
  - Functions and variables: `lowercase_with_underscores` or `lowercamelcase`
  - Types and modules: `UpperCamelCase`
  - Constants: `UPPERCASE_WITH_UNDERSCORES`
  - Type parameters: Single uppercase letters (T, S, etc.)

- **Indentation:** 4 spaces (no tabs)

- **Line Length:** Aim for 92 characters, max 120

- **Comments:**
  - Use `#` for single-line comments
  - Use docstrings (""") for function documentation

- **Docstrings:**
  ```julia
  """
      function_name(arg1, arg2)

  Brief description of what the function does.

  # Arguments
  - `arg1::Type`: Description of arg1
  - `arg2::Type`: Description of arg2

  # Returns
  - `ReturnType`: Description of return value

  # Examples
  ```julia
  julia> function_name(1, 2)
  3
  ```
  """
  function function_name(arg1, arg2)
      # implementation
  end
  ```

### 4. Testing

Julia uses the built-in `Test` module:

**test/runtests.jl:**
```julia
using Test
using YourPackageName

@testset "YourPackageName Tests" begin
    @testset "Feature 1" begin
        @test function_to_test(input) == expected_output
    end

    @testset "Feature 2" begin
        @test another_function(input) ≈ expected_output atol=1e-10
    end
end
```

**Running Tests:**
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
# or
julia --project=. test/runtests.jl
```

### 5. Dependencies Management

**Adding Dependencies:**
```bash
julia --project=. -e 'using Pkg; Pkg.add("PackageName")'
```

**Updating Dependencies:**
```bash
julia --project=. -e 'using Pkg; Pkg.update()'
```

**Installing Project Dependencies:**
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Development Workflow

### Setting Up Development Environment

1. **Install Julia:** Ensure Julia 1.0+ is installed
2. **Activate Project:**
   ```bash
   julia --project=.
   ```
3. **Install Dependencies:**
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

### Daily Development Workflow

1. **Start Julia REPL with project:**
   ```bash
   julia --project=.
   ```

2. **Load package in development mode:**
   ```julia
   using Revise  # Auto-reloads code changes
   using YourPackageName
   ```

3. **Make changes to source files** - Revise will automatically reload

4. **Run tests:**
   ```julia
   using Pkg
   Pkg.test()
   ```

### Performance Optimization

Julia-specific performance tips:

- Use type annotations for clarity, not speed (compiler infers types)
- Avoid global variables; use `const` for global constants
- Use `@time` and `@benchmark` (from BenchmarkTools.jl) for profiling
- Write type-stable functions
- Use `@inbounds` and `@simd` when safe to do so
- Prefer column-major order for array access

## Git Workflow

### Branch Strategy

- **Main Branch:** Stable, production-ready code
- **Feature Branches:** Named `claude/claude-md-*` for AI assistant work
- **Development:** Work on designated feature branches

### Commit Guidelines

- Use descriptive commit messages
- Format: `<type>: <description>`
  - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Examples:
  - `feat: add matrix multiplication function`
  - `fix: correct off-by-one error in indexing`
  - `docs: update README with installation instructions`

### Push Protocol

```bash
git add .
git commit -m "feat: descriptive message"
git push -u origin claude/claude-md-mi0bulaexf0g5nv4-015mGhm324Krpiqwg4a5dnSD
```

## AI Assistant Conventions

### Code Generation Guidelines

1. **Always Create Tests:** When implementing new functionality, create corresponding tests

2. **Use Type Annotations:** Help with code clarity:
   ```julia
   function process_data(data::Vector{Float64})::Matrix{Float64}
       # implementation
   end
   ```

3. **Document Everything:** Use docstrings for all exported functions

4. **Follow Julia Idioms:**
   - Use broadcasting (`.` operator) for vectorized operations
   - Use multiple dispatch appropriately
   - Prefer immutability when possible
   - Use `!` suffix for mutating functions

5. **Error Handling:**
   ```julia
   function safe_divide(a, b)
       b == 0 && throw(DivideError())
       return a / b
   end
   ```

### File Operations

When creating new files:

1. **Source Files:** Place in `src/` directory
2. **Test Files:** Place in `test/` directory, name as `test_*.jl`
3. **Examples:** Place in `examples/` directory
4. **Documentation:** Place in `docs/src/` directory

### Package Development Checklist

When developing this repository into a full package:

- [ ] Create `Project.toml` with package metadata
- [ ] Create `src/` directory with main module file
- [ ] Create `test/` directory with `runtests.jl`
- [ ] Add `.gitignore` for Julia (Manifest.toml, *.jl.cov, etc.)
- [ ] Set up documentation with Documenter.jl
- [ ] Add CI/CD (GitHub Actions recommended)
- [ ] Add LICENSE file
- [ ] Create comprehensive README.md

## Common Julia Commands Reference

```bash
# Start Julia REPL with project
julia --project=.

# Run script
julia --project=. script.jl

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Add package
julia --project=. -e 'using Pkg; Pkg.add("PackageName")'

# Update packages
julia --project=. -e 'using Pkg; Pkg.update()'

# Check package status
julia --project=. -e 'using Pkg; Pkg.status()'

# Build project
julia --project=. -e 'using Pkg; Pkg.build()'

# Precompile packages
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## Resources

- [Julia Documentation](https://docs.julialang.org/)
- [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [Julia Package Development](https://pkgdocs.julialang.org/v1/)
- [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [Pkg.jl Documentation](https://pkgdocs.julialang.org/)

## Notes for AI Assistants

1. **Always verify Julia version compatibility** when suggesting code or packages

2. **Use Julia-specific patterns:**
   - Multiple dispatch over if-else type checking
   - Broadcasting over loops where appropriate
   - Native Julia types over custom implementations when possible

3. **Consider Performance:** Julia is designed for high-performance computing
   - Avoid type instability
   - Pre-allocate arrays when possible
   - Use views instead of copies when appropriate

4. **Testing is Critical:** Julia's dynamic nature makes comprehensive testing essential

5. **Documentation:** Use Julia's docstring system extensively

6. **When in doubt:** Check if there's a standard library or registered package that already implements the functionality

## Project-Specific Notes

This repository is currently minimal. When beginning development:

1. Decide if this will be a package or application
2. Create appropriate `Project.toml`
3. Establish directory structure
4. Set up testing framework
5. Define coding standards for the team

---

*This CLAUDE.md file should be updated as the project evolves and new conventions are established.*
