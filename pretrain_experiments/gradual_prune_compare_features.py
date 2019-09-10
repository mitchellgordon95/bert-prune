from univa_grid import TaskRunner
import subprocess
import os
import math

def compare_features(sparsity):
    model_name = f"gradual_prune_{int(sparsity*100)}"
    out_f = open(f'models/pretrain/{model_name}/features_compare_0.txt', 'w+')
    subprocess.call([
        'python', 'checkpoint_utils/compare_features.py', f'models/pretrain/gradual_prune_0/features.json', f'models/pretrain/{model_name}/features.json'], stdout=out_f)

task_runner = TaskRunner()
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(compare_features, sparsity)
