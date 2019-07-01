import subprocess
from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict

def train_downstream(sparsity):
    init_model_name = f"gradual_prune_{int(sparsity*100)}_sign_ticket"
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'STS-B', 'MRPC', 'RTE']:
        for lr in ['2e-5','3e-5','4e-5','5e-5',]:
            for epoch in range(4):
                train(task, init_model_name, epoch, lr)
                eval_(task, init_model_name, lr)

task_runner = TaskRunner()
for sparsity in [0, .4, .5, .6, .7, .8, .9]: # -t 1-7 for univa grid engine
   task_runner.do_task(train_downstream, sparsity)
