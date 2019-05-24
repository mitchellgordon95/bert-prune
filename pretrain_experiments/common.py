from collections import namedtuple
import subprocess
import os

BERT_BASE_DIR = "uncased_L-12_H-768_A-12_prunable"
OUTPUT_DIR = "models"
TRAIN_128 = "data/pretrain_examples_len_128/train/*"
DEV_128 = "data/pretrain_examples_len_128/dev/*"

class SparsityHParams(
        namedtuple('SparsityHParams', [
            'initial_sparsity',
            'target_sparsity',
            'sparsity_function_end_step',
            'end_pruning_step'])):
    """See:
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning"""
    def __str__(self):
        return (f'initial_sparsity={self.initial_sparsity},'
                f'target_sparsity={self.target_sparsity},'
                f'sparsity_function_end_step={self.sparsity_function_end_step},'
                f'end_pruning_step={self.end_pruning_step}')

def pretrain(model_name, input_file, num_train_steps, sparsity_hparams: SparsityHParams):
    _run_pretraining(model_name, input_file, True, False, num_train_steps, 0, sparsity_hparams)

def pretrain_eval(model_name, input_file, max_eval_steps):
    _run_pretraining(model_name, input_file, False, True, 0, max_eval_steps, None)

def _run_pretraining(
        model_name,
        input_file,
        do_train,
        do_eval,
        num_train_steps,
        max_eval_steps,
        sparsity_hparams: SparsityHParams,
        ):
    train_args = ["--do_train=True", "--num_train_steps", str(num_train_steps)] if do_train else []
    eval_args = ["--do_eval=True", "--max_eval_steps", str(max_eval_steps)] if do_eval else []
    sparsity_args = ['--pruning_hparams', str(sparsity_hparams)] if sparsity_hparams else []
    subprocess.call([
        "python", "run_pretraining.py",
        "--input_file", input_file,
        "--output_dir", os.path.join(OUTPUT_DIR, model_name),
        "--bert_config_file", f"{BERT_BASE_DIR}/bert_config.json",
        "--init_checkpoint", f"{BERT_BASE_DIR}/bert_model.ckpt",
        "--train_batch_size", "32",
        "--max_seq_length", "128",
        "--max_predictions_per_seq", "20",
        "--num_warmup_steps", "10",
        "--learning_rate", "2e-5"] + train_args + eval_args + sparsity_args
    )
