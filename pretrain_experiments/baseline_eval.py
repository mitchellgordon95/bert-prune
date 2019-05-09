import subprocess
from univa_grid import TaskRunner

BERT_BASE_DIR = "uncased_L-12_H-768_A-12_prunable"
OUTPUT_DIR = "models/uncased_base_baseline"

subprocess.call([
    "python", "run_pretraining.py",
    "--input_file", "data/pretrain_examples_len_128/tf_examples*.tfrecord",
    "--output_dir", f"{OUTPUT_DIR}",
    "--do_train", "True",
    "--do_eval", "True",
    "--bert_config_file", f"{BERT_BASE_DIR}/bert_config.json",
    "--init_checkpoint", f"{BERT_BASE_DIR}/bert_model.ckpt",
    "--train_batch_size", "32",
    "--max_seq_length", "128",
    "--max_predictions_per_seq", "20",
    "--num_train_steps", "200",
    "--max_eval_steps", "100", # TODO (mitchg) - the default eval steps is 100. I feel like we definitely want more...
    "--num_warmup_steps", "10",
    "--learning_rate", "2e-5",
])
