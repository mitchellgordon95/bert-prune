from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict, MODELS_DIR

def train_downstream(sparsity):
    model_name = f"gradual_prune_{int(sparsity*100)}_randomized_mask"
    init_model_dir = f"{MODELS_DIR}/pretrain/{model_name}"
    # This is the order from least training data to most training data
    for task in [ 'CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']:
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            for epoch in range(4):
                train(task, init_model_dir, model_name, epoch, lr)
                eval_(task, model_name, lr)

            eval_(task, model_name, lr, use_train_data=True)

task_runner = TaskRunner()
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]: # -t 1-10 for univa grid engine
   task_runner.do_task(train_downstream, sparsity)
