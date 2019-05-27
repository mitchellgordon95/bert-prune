import subprocess
from univa_grid import TaskRunner
# from downstream_experiments.common import
import os

BERT_BASE_DIR = "uncased_L-12_H-768_A-12_prunable"
MODELS_DIR = "models"
GLUE_DIR = "data/glue_data"

def _run_classifier(
        task_name,
        init_model_name,
        do_train,
        do_eval,
        do_predict,
        num_train_epochs,
        lr="2e-5",
        ):
    train_args = ["--do_train=True", "--num_train_epochs", str(num_train_epochs)] if do_train else []
    subprocess.call([
        "python", "run_classifier.py",
        "--task_name", task_name,
        "--data_dir", f'{GLUE_DIR}/{task_name}',
        "--output_dir", os.path.join(MODELS_DIR, task_name, f'{init_model_name}_lr_{lr}'),
        "--bert_config_file", f"{BERT_BASE_DIR}/bert_config.json",
        "--init_checkpoint", f"{MODELS_DIR}/pretrain/{init_model_name}",
        "--vocab_file", f"{BERT_BASE_DIR}/vocab.txt",
        "--train_batch_size", "32",
        "--max_seq_length", "128",
        "--do_eval", str(do_eval),
        "--do_predict", str(do_predict),
        "--learning_rate", "2e-5"] + train_args
    )

def train(task_name, init_model_name, num_train_epochs, lr):
    _run_classifier(task_name, init_model_name, True, False, False, num_train_epochs, lr)

def eval_(task_name, init_model_name):
    _run_classifier(task_name, init_model_name, False, True, False, 0)

def predict(task_name, init_model_name):
    _run_classifier(task_name, init_model_name, False, False, True, 0)

def train_downstream(sparsity):
    init_model_name = f"gradual_prune_{int(sparsity*100)}"
    # TODO (mitchg) we're missing some tasks because run_classifier doesn't support them
    # QQP, QNLI, SST-2, STS-B, RTE
    for task in ['MRPC', 'CoLA', 'MNLI', 'XNLI']:
        for lr in ['2e-5','3e-5','4e-5','5e-5',]:
            for epoch in range(10):
                train(task, init_model_name, epoch, lr)
                eval_(task, init_model_name)

task_runner = TaskRunner()
for sparsity in [0, .4, .5, .6, .7, .8, .9]: # -t 1-7 for univa grid engine
   task_runner.do_task(train_downstream, sparsity)
