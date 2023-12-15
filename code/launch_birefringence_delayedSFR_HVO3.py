#!/bin/bash

#SBATCH --job-name=array
#SBATCH --account=kicp
#SBATCH --output=logs/log_delayedSFR_HVO3.out
#SBATCH --error=logs/log_delayedSFR_HVO3.err
#SBATCH --time=12:00:00
#SBATCH --partition=kicp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

nvidia-smi

# Add lines here to run your computation on each job
cd /home/tcallister/repositories/stochastic-birefringence/code/
conda activate stochastic-birefringence
python run_birefringence_delayedSFR_HVO3.py
