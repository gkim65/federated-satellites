#!/bin/bash

source /nfs-share/grk27/miniconda3/bin/activate fed_sats

srun -w mauao -c 10 --gres=gpu:1 --partition=interactive --pty bash 
srun -w ruapehu -c 20 --gres=gpu:0 --partition=interactive --pty bash

ngongotaha