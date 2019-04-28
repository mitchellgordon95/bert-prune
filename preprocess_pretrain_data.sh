#! /bin/bash
#$ -cwd
#$ -V
#$ -l num_proc=2,h_rt=10:00:00
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
python preprocess_pretrain_data.py
