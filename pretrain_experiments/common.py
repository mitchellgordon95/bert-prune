from collections import namedtuple

BERT_BASE_DIR = "uncased_L-12_H-768_A-12_prunable"
OUTPUT_DIR = "models"

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

def run_pretrain_base(
        model_name,
        do_train,
        do_eval,
        num_train_steps,
        max_eval_steps,
        sparsity_hparams: SparsityHParams,
        ):
    subprocess.call([
        "python", "run_pretraining.py",
        "--input_file", "data/pretrain_examples_len_128/tf_examples*.tfrecord",
        "--output_dir", os.path.join(OUTPUT_DIR,model_name),
        "--do_train", str(do_train),
        "--do_eval", str(do_eval),
        "--bert_config_file", f"{BERT_BASE_DIR}/bert_config.json",
        "--init_checkpoint", f"{BERT_BASE_DIR}/bert_model.ckpt",
        "--train_batch_size", "32",
        "--max_seq_length", "128",
        "--max_predictions_per_seq", "20",
        "--num_train_steps", str(num_train_steps),
        "--max_eval_steps", str(max_eval_steps), # TODO (mitchg) - the default eval steps is 100. I feel like we definitely want more...
        "--num_warmup_steps", "10",
        "--learning_rate", "2e-5",
        '--pruning_hparams', str(sparsity_hparams)
    ])
