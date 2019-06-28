#! /bin/bash
#$ -cwd
#$ -V
#$ -l gpu=1,h_rt=96:00:00,num_proc=2,mem_free=3G
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
#$ -q gpu.q

# Note: Titan XP gpus have the most memory on the grid (12 GB)
# However, with gradient checkpointing we can get away with using basically any GPU

ml load cuda90/toolkit
export PYTHONPATH=.
python $1 "${@:2}"
