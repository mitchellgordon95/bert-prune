from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict, MODELS_DIR
import subprocess

def extract_features(sparsity):
    model_name = f"gradual_prune_{int(sparsity*100)}_no_mask"
    # This is the order from least training data to most training data
    for task in ['CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']:
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            subprocess.call([
                'python', 'extract_features.py',
                '--input_file', f'data/features_dev.txt',
                '--output_file', f'models/{task}/{model_name}_lr_{lr}/features.json',
                '--bert_config_file', 'uncased_L-12_H-768_A-12/bert_config.json',
                '--layers', '-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12',
                '--vocab_file', 'uncased_L-12_H-768_A-12/vocab.txt',
                '--init_checkpoint', f'models/{task}/{model_name}_lr_{lr}',
                '--do_lower_case', 'True',
                '--max_seq_length', '128'])

task_runner = TaskRunner()
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(extract_features, sparsity)
