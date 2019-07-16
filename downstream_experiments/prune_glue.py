from univa_grid import TaskRunner
from downstream_experiments.common import train, eval_, predict, MODELS_DIR, BERT_BASE_DIR
from experiments_common import SparsityHParams

def train_downstream(sparsity):
    model_name = f"downstream_prune_{int(sparsity*100)}"
    # This is the order from least training data to most training data
    for task in [ 'RTE',  'MRPC', 'STS-B',  'CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']:
        for epoch in range(3):
            train(task, BERT_BASE_DIR, model_name, epoch, "2e-5")
            eval_(task, model_name, "2e-5")

        # Prunes weights in one shot
        train(task, BERT_BASE_DIR, model_name, 3, "2e-5", sparsity_hparams=SparsityHParams(
                initial_sparsity=sparsity,
                target_sparsity=sparsity,
                sparsity_function_end_step=1,
                ))
        eval_(task, model_name, "2e-5")

task_runner = TaskRunner()
for sparsity in [0, .3, .4, .5, .6, .7, .8, .9]: # -t 1-8 for univa grid engine
   task_runner.do_task(train_downstream, sparsity)
