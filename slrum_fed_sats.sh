#!/bin/bash
#SBATCH --cpus-per-task 30
#SBATCH --gres=gpu:0
#SBATCH --job-name=grace_practice
#SBATCH --partition=normal
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G

source /nfs-share/grk27/miniconda3/bin/activate fed_sats

srun python /nfs-share/grk27/Documents/more_tests2/federated-satellites/server.py