#! /bin/bash
#$ -cwd
#$ -V
#$ -l gpu=1,h_rt=160:00:00,num_proc=2,mem_free=3G
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
#$ -q gpu.q@@2080

# Note: Titan XP gpus have the most memory on the grid (12 GB)
# However, with gradient checkpointing we can get away with using basically any GPU

ml load cuda10.0/toolkit
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi
export PYTHONPATH=.
python $1 "${@:2}"
