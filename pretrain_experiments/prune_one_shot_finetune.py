import subprocess
from univa_grid import TaskRunner
from pretrain_experiments.common import TRAIN_128, DEV_128, pretrain, pretrain_eval, SparsityHParams
import os

def prune_finetune(sparsity):
    # TODO (mitchg) - also finetune on max_seq_len=500
    model_name = f"base_prune_{int(sparsity*100)}"
    for step in range(20):
        pretrain(
            input_file=TRAIN_128,
            model_name=model_name,
            num_train_steps=step*5000,
            sparsity_hparams=SparsityHParams(
                initial_sparsity=sparsity,
                target_sparsity=sparsity,
                sparsity_function_end_step=1,
                end_pruning_step=200,
                )
        )
        # TODO (mitchg) what's the right number here?
        pretrain_eval(model_name=model_name, input_file=DEV_128, max_eval_steps=1000)


task_runner = TaskRunner()
for sparsity in [.4, .5, .6, .7, .8, .9]: # -t 1-6 for univa grid engine
   task_runner.do_task(prune_finetune, sparsity)
