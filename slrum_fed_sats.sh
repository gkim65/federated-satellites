#!/bin/bash
#SBATCH --cpus-per-task 20
#SBATCH --gres=gpu:0
#SBATCH --job-name=grace_practice
#SBATCH --partition=normal
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G
#SBATCH -w ruapehu

source /nfs-share/grk27/miniconda3/bin/activate fed_sats

srun python /nfs-share/grk27/Documents/federated-satellites/server.py