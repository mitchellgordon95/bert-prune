from univa_grid import TaskRunner
import subprocess
import os
import math

def extract_features(sparsity):
    model_name = f"gradual_prune_{int(sparsity*100)}"
    subprocess.call([
        'python', 'extract_features.py',
        '--input_file', f'data/features_dev.txt',
        '--output_file', f'models/pretrain/{model_name}/features.json',
        '--bert_config_file', 'uncased_L-12_H-768_A-12/bert_config.json',
        '--layers', '-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12',
        '--vocab_file', 'uncased_L-12_H-768_A-12/vocab.txt',
        '--init_checkpoint', f'models/pretrain/{model_name}',
        '--do_lower_case', 'True',
        '--max_seq_length', '128'])

task_runner = TaskRunner()
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(extract_features, sparsity)
