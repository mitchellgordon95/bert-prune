from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict, MODELS_DIR
from experiments_common import SparsityHParams
import shutil

def train_downstream(task):
    lr = '5e-5'
    for starting_epoch in [0, 1, 2]: #[3, 6, 9, 12]:
        if starting_epoch == 0:
            init_checkpoint = f'models/pretrain/burned_in'
        else:
            init_checkpoint = f'models/{task}/trained_downstream_lr_{lr}_epoch_{starting_epoch}'
        model_name = f'trained_downstream_lr_{lr}_epoch_{starting_epoch}_pruned_60'
        for epoch in [1, 2]:
            eval_(task, model_name, lr)
            train(task, init_checkpoint, model_name, epoch, lr, sparsity_hparams=SparsityHParams(
                    initial_sparsity=0.6,
                    target_sparsity=0.6,
                    sparsity_function_end_step=1,
                    end_pruning_step=-1,
                    ))
        eval_(task, model_name, lr)
        eval_(task, model_name, lr, use_train_data=True)

task_runner = TaskRunner()
for task in ['CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']:
   task_runner.do_task(train_downstream, task)
