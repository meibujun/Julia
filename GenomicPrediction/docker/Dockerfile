# Dockerfile for GenomicPrediction.jl

# Use the official Julia base image
FROM julia:1.11

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY ./GenomicPrediction/Project.toml /app/Project.toml
COPY ./GenomicPrediction/Manifest.toml /app/Manifest.toml

# Instantiate the environment to download and precompile dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Copy the rest of the source code
COPY ./GenomicPrediction/src /app/src

# Set the entrypoint to run a Julia REPL in the project environment
CMD ["julia", "--project=."]
