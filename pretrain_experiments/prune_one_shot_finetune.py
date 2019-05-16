import subprocess
from univa_grid import TaskRunner
from pretrain_experiments.common import run_pretrain_base, SparsityHParams
import os

def prune_finetune(sparsity):
    # TODO (mitchg) - also finetune on max_seq_len=50
    run_pretrain_base(
        model_name=f"base_prune_{int(sparsity*100)}",
        do_train=True
        do_eval=False,
        num_train_steps=10000,
        max_eval_steps=0,
        sparsity_hparams=SparsityHParams(
            initial_sparsity=sparsity,
            target_sparsity=sparsity,
            sparsity_function_send_step=1,
            end_pruning_step=200,
            )
    )


task_runner = TaskRunner()
for sparsity in [.4, .5, .6, .7, .8, .9]: # -t 1-6 for univa grid engine
   task_runner.do_task(prune_finetune, sparsity)
