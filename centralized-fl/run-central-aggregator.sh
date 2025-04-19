#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=256GB
#SBATCH --time=1-00:00
#SBATCH --partition=livi
#SBATCH --mail-user=sendena@myumanitoba.ca
#SBATCH --mail-type=ALL

# export HF_HOME=~/projects/def-cjhuofw-ab/sendena/hf-cache
#export RAY_DEDUP_LOGS=0

# module load cuda/12.4.1 arch/avx2 gcc/13.2.0 python/3.11.11

# source ~/env/env_flwr/bin/activate

pip install -e .

strategies=(
    FedAvg
    FedYogi
    FedAvgM
    FedMedian
    Krum
    MultiKrum
)

mkdir -p log/${SLURM_JOB_ID}

for strategy in "${strategies[@]}"; do
    echo "Starting ${strategy} run at: `date`"
    flwr run . --run-config "strategy=\"$strategy\""
    echo "${strategy} finished with exit code $? at: `date`"
done
