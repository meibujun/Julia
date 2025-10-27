# Dockerfile for GenomicPrediction.jl
# =====================================
# This Dockerfile creates a container image with a fully configured environment
# for running the GenomicPrediction.jl package.

# Use the official Julia base image, specifying the version
FROM julia:1.11.6

# --- Metadata ---
LABEL maintainer="The GenomicPrediction.jl Development Team"
LABEL description="A comprehensive software package for genomic prediction in Julia."

# --- Environment Setup ---
# Set the working directory
WORKDIR /app

# Copy the entire project into the container's working directory
# This includes source, tests, examples, etc.
COPY . .

# --- Julia Package Management ---
# Activate the project environment and instantiate/precompile all dependencies.
# This ensures that when the container starts, all packages are ready to go.
RUN julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# --- Networking ---
# Expose ports for potential web UIs, as described in the design document.
# 8080 for a Dash app, 1234 for Pluto notebooks.
EXPOSE 8080
EXPOSE 1234

# --- Entrypoint ---
# When the container is run, it will start a Julia REPL in the project environment.
# Users can then run `using GenomicPrediction` and execute commands.
CMD ["julia", "--project=."]
