FROM julia:1.11-buster

WORKDIR /opt/GenomicPrediction
COPY .. /opt/GenomicPrediction

RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

EXPOSE 8050 1234

CMD ["julia", "--project=.", "examples/scripts/basic_workflow.jl"]
