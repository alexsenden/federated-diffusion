#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --time=2-00:00
#SBATCH --partition=livi
#SBATCH --mail-user=sendena@myumanitoba.ca
#SBATCH --mail-type=ALL

export HF_HOME=~/projects/def-cjhuofw-ab/sendena/hf-cache

module load cuda/12.4.1 arch/avx2 gcc/13.2.0 python/3.11.11

source ~/env/env_flwr/bin/activate

# pip install -r requirements.txt

echo "Starting training run at: `date`"
python train_centralized_model.py
echo "training finished with exit code $? at: `date`"
