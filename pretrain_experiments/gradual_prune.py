import subprocess
from univa_grid import TaskRunner
from pretrain_experiments.common import TRAIN_128, DEV_128, pretrain, pretrain_eval, SparsityHParams
import os

def gradual_prune(sparsity):
    # TODO (mitchg) - also finetune on max_seq_len=500
    model_name = f"gradual_prune_{int(sparsity*100)}"
    for step in range(20):
        pretrain(
            input_file=TRAIN_128,
            model_name=model_name,
            num_train_steps=5000*step,
            sparsity_hparams=SparsityHParams(
                initial_sparsity=0,
                target_sparsity=sparsity,
                sparsity_function_end_step=10000,
                end_pruning_step=-1,
                )
        )
        # TODO (mitchg) what's the right number here?
        pretrain_eval(model_name=model_name, input_file=DEV_128, max_eval_steps=2000)


task_runner = TaskRunner()
for sparsity in [0, .4, .5, .6, .7, .8, .9]: # -t 1-6 for univa grid engine
   task_runner.do_task(gradual_prune, sparsity)
