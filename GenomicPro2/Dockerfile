# GenomicPro2 Docker Image
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM julia:1.10 as builder

# Set working directory
WORKDIR /app

# Copy Project files
COPY GenomicPro2/Project.toml GenomicPro2/Manifest.toml ./

# Install Julia packages
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

# Copy source code
COPY GenomicPro2/src ./src
COPY GenomicPro2/test ./test
COPY GenomicPro2/examples ./examples
COPY GenomicPro2/docs ./docs

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM julia:1.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV JULIA_NUM_THREADS=auto
ENV JULIA_DEPOT_PATH=/opt/julia
ENV GENOMICPRO_HOME=/app

# Create application directory
WORKDIR /app

# Copy from builder
COPY --from=builder /root/.julia /root/.julia
COPY --from=builder /app /app

# Install Python package (optional)
COPY python /app/python
RUN pip3 install --no-cache-dir -e /app/python

# Expose API port
EXPOSE 8080

# Set entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["julia"]
