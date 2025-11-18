#!/bin/bash
# Docker entrypoint script for GenomicPro2

set -e

echo "=================================================="
echo "  GenomicPro2 Docker Container"
echo "=================================================="
echo ""
echo "Julia version: $(julia --version)"
echo "Threads: ${JULIA_NUM_THREADS:-auto}"
echo "Working directory: $(pwd)"
echo ""

# Activate Julia project
cd /app
julia -e 'using Pkg; Pkg.activate("."); Pkg.status()'

# Execute command
exec "$@"
