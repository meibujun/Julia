#!/bin/bash
#SBATCH --job-name=genomic_prediction
#SBATCH --output=logs/genomic_prediction_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load julia/1.11

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs

julia --project=. -e 'using GenomicPrediction, Random; seed = parse(Int, get(ENV, "SLURM_JOB_ID", "1")); Random.seed!(seed); dataset = simulate_genomic_data(500, 1000; h2=0.6); pipeline = default_workflow(dataset); run_autogs(pipeline, dataset); println(pipeline.results)'
