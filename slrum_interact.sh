#!/bin/bash
srun -w mauao -c 10 --gres=gpu:1 --partition=interactive bash 
srun -w ruapehu -c 10 --gres=gpu:0 --partition=interactive bash 