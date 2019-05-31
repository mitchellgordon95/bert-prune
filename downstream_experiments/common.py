import subprocess
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
        num_train_epochs=0,
        lr="2e-5",
        ):
    train_args = ["--do_train=True", "--num_train_epochs", str(num_train_epochs)] if do_train else []
    eval_args = ["--do_eval=True"] if do_eval else []
    predict_args = ["--do_predict=True"] if do_predict else []
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
        "--learning_rate", "2e-5"] + train_args + eval_args + predict_args
    )

def train(task_name, init_model_name, num_train_epochs, lr):
    _run_classifier(task_name, init_model_name, do_train=True, do_eval=False, do_predict=False, num_train_epochs=num_train_epochs, lr=lr)

def eval_(task_name, init_model_name):
    _run_classifier(task_name, init_model_name, do_train=False, do_eval=True, do_predict=False)

def predict(task_name, init_model_name):
    _run_classifier(task_name, init_model_name, do_train=False, do_eval=False, do_predict=True)
