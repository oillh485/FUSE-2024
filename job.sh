#!/bin/bash

#SBATCH -N 150
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -J henrys_first_job
#SBATCH --mail-user=henry.g.o@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -A m3980_g
#SBATCH --ntasks-per-node=4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err


### Load modules
ml conda/Mambaforge-23.1.0-1

# activate environment
mamba activate TestDev

srun -n 600 -c 32 --gpus-per-task 1 --mem-per-gpu=100G -G 1 python my_python_script.py

wait