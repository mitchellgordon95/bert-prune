import subprocess
from univa_grid import TaskRunner
from pretrain_experiments.common import TRAIN_128, DEV_128, pretrain, pretrain_eval
from experiments_common import SparsityHParams
import os

def gradual_prune(sparsity):
    # TODO (mitchg) - also finetune on max_seq_len=500
    model_name = f"gradual_prune_{int(sparsity*100)}_less_embed"
    for step in range(1, 20):
        pretrain(
            input_file=TRAIN_128,
            model_name=model_name,
            num_train_steps=5000*step,
            sparsity_hparams=SparsityHParams(
                initial_sparsity=0,
                target_sparsity=sparsity,
                sparsity_function_end_step=10000,
                end_pruning_step=-1,
                weight_sparsity_map='[.*embeddings.*:0.4]'
                )
        )
        # Note: max_eval_steps is the number of batches we process
        # default batch size is 8
        pretrain_eval(model_name=model_name, input_file=DEV_128, max_eval_steps=2000)


task_runner = TaskRunner()
for sparsity in [.5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(gradual_prune, sparsity)
