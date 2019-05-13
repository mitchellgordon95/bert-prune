import subprocess
from univa_grid import TaskRunner
import os

BERT_BASE_DIR = "uncased_L-12_H-768_A-12_prunable"
OUTPUT_DIR = "models"

def prune_finetune(sparsity):
    subprocess.call([
        "python", "run_pretraining.py",
        "--input_file", "data/pretrain_examples_len_128/tf_examples*.tfrecord",
        "--output_dir", os.path.join(OUTPUT_DIR, f"gradual_prune_{int(sparsity*100)}"),
        "--do_train", "True",
        "--do_eval", "True",
        "--bert_config_file", os.path.join(BERT_BASE_DIR, "bert_config.json"),
        "--init_checkpoint", os.path.join(BERT_BASE_DIR, "bert_model.ckpt"),
        "--train_batch_size", "32",
        # TODO (mitchg) - also finetune on max_seq_len=50
        "--max_seq_length", "128",
        "--max_predictions_per_seq", "20",
        "--num_train_steps", "10000",
        "--max_eval_steps", "100", # TODO (mitchg) - the default eval steps is 100. I feel like we definitely want more...
        "--num_warmup_steps", "10",
        "--learning_rate", "2e-5",
        '--pruning_hparams', f'initial_sparsity=0,target_sparsity={sparsity},sparsity_function_end_step=10000,end_pruning_step=-1'
    ])


task_runner = TaskRunner()
for sparsity in [.4, .5, .6, .7, .8, .9]: # -t 1-6 for univa grid engine
   task_runner.do_task(prune_finetune, sparsity)
