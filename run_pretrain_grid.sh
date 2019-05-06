#! /bin/bash
#$ -cwd
#$ -V
#$ -l gpu=1,h_rt=24:00:00,num_proc=2,mem_free=3G
#$ -j y
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
#$ -q gpu.q@@titanxp

# Note: Titan XP gpus have the most memory on the grid (12 GB)

ml load cuda90/toolkit

BERT_BASE_DIR="uncased_L-12_H-768_A-12_prunable"
python run_pretraining.py \
  --input_file=data/pretrain_examples_len_128/tf_examples*.tfrecord \
  --output_dir=models/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
