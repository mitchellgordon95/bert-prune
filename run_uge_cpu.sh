#! /bin/bash
#$ -cwd
#$ -V
#$ -l h_rt=10:00:00,num_proc=4,mem_free=5G
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com

export PYTHONPATH=.
python $1 "${@:2}"
