#!/bin/zsh
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --job-name=grace_practice

source /nfs-share/grk27/miniconda3/bin/activate fed_sats

srun python /nfs-share/grk27/Documents/federated-satellites/server.py