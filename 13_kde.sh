#!/bin/bash
#SBATCH --job-name=kde_$1
#SBATCH --output=logs/kde_$1.out
#SBATCH --error=logs/kde_$1.err
#SBATCH --gres=gpu:1

module load python/3.13
# conda activate my_env

python 13_pytorch_KDE_eval_valuesGathering.py --case $1
