from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict, MODELS_DIR
from experiments_common import SparsityHParams
import shutil

def train_downstream(task):
    lr = '5e-5'
    init_checkpoint = "models/pretrain/burned_in"
    model_name = f"trained_downstream"
    for epoch in range(1, 13):
        train(task, init_checkpoint, model_name, epoch, lr,
            sparsity_hparams=SparsityHParams(
                    initial_sparsity=0,
                    target_sparsity=0,
                    sparsity_function_end_step=1,
                    end_pruning_step=-1,
                    )
            )
        shutil.copytree(f'models/{task}/trained_downstream_lr_{lr}', f'models/{task}/trained_downstream_lr_{lr}_epoch_{epoch}')

task_runner = TaskRunner()
for task in ['CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']:
   task_runner.do_task(train_downstream, task)
