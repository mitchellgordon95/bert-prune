from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict, MODELS_DIR
from experiments_common import SparsityHParams

def train_downstream(sparsity):
    init_checkpoint = "models/pretrain/gradual_prune_0"
    model_name = f"gradual_prune_0_downstream_prune_{int(sparsity*100)}"
    # This is the order from least training data to most training data
    for task in ['CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']: 
        for lr in ['2e-5', '3e-5', '4e-5', '5e-5']: #
            for epoch in range(4):
                train(task, init_checkpoint, model_name, epoch, lr,
                    sparsity_hparams=SparsityHParams(
                            initial_sparsity=0,
                            target_sparsity=0,
                            sparsity_function_end_step=1,
                            end_pruning_step=-1,
                            )
                    )
                eval_(task, model_name, lr)

            # Prunes weights in one shot
            for epoch in [3.1, 4, 5, 6, 7, 8]:
                train(task, init_checkpoint, model_name, epoch, lr, sparsity_hparams=SparsityHParams(
                        initial_sparsity=sparsity,
                        target_sparsity=sparsity,
                        sparsity_function_end_step=1,
                        end_pruning_step=-1,
                        ))
                eval_(task, model_name, lr)

            if sparsity in [0.8, 0.9]:
                for epoch in [9, 10]:
                    train(task, init_checkpoint, model_name, epoch, lr, sparsity_hparams=SparsityHParams(
                            initial_sparsity=sparsity,
                            target_sparsity=sparsity,
                            sparsity_function_end_step=1,
                            end_pruning_step=-1,
                            ))
                    eval_(task, model_name, lr)

            eval_(task, model_name, lr, use_train_data=True)

task_runner = TaskRunner()
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(train_downstream, sparsity)
