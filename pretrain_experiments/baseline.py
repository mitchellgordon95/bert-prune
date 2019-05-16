import subprocess
from univa_grid import TaskRunner
from pretrain_experiments.common import run_pretrain_base, SparsityHParams

OUTPUT_DIR = "models/uncased_base_baseline"

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
