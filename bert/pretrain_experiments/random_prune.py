import subprocess
from univa_grid import TaskRunner
from pretrain_experiments.common import TRAIN_128, DEV_128, pretrain, pretrain_eval
import os
from checkpoint_utils.random_masks import random_masks

def random_prune(sparsity):
    model_name = f"burned_in_random_prune_{int(sparsity*100)}"
    if not os.path.exists(f'models/pretrain/{model_name}'):
        random_masks(f'models/pretrain/burned_in', sparsity)

    for step in range(20):
        pretrain(
            input_file=TRAIN_128,
            model_name=model_name,
            num_train_steps=5000*step,
            sparsity_hparams=None,
        )
        # Note: max_eval_steps is the number of batches we process
        # default batch size is 8
        pretrain_eval(model_name=model_name, input_file=DEV_128, max_eval_steps=2000)

task_runner = TaskRunner()
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(random_prune, sparsity)
