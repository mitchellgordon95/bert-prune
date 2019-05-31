#! /bin/bash
#$ -cwd
#$ -V
#$ -l gpu=1,h_rt=48:00:00,num_proc=2,mem_free=3G
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
#$ -q gpu.q@@titanxp

# Note: Titan XP gpus have the most memory on the grid (12 GB)

ml load cuda90/toolkit
export PYTHONPATH=.
python $1 "${@:2}"
