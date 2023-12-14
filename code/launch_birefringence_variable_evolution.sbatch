#!/bin/bash

#SBATCH --job-name=array
#SBATCH --account=kicp
#SBATCH --output=logs/log.out
#SBATCH --error=logs/log.err
#SBATCH --time=12:00:00
#SBATCH --partition=kicp-gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=20G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

nvidia-smi

# Add lines here to run your computation on each job
cd /home/tcallister/repositories/stochastic-birefringence/code/
conda activate stochastic-birefringence-cuda
python run_birefringence_variable_evolution.py
