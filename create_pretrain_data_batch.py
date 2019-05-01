from univa_grid import TaskRunner
import subprocess

def create_pretrain(task_id):
    subprocess.call([
        'python', 'create_pretraining_data.py',
        '--input_file', f'data/pretrain_sentencized/pretrain_sentencized_{task_id}.txt',
        '--output_file', f'data/pretrain_examples/tf_examples_{task_id}.tfrecord',
        '--vocab_file', 'uncased_L-12_H-768_A-12/vocab.txt',
        '--do_lower_case', 'True',
        '--max_seq_length', '128',
        '--max_predictions_per_seq', '20',
        '--masked_lm_prob', '0.15',
        '--random_seed', '12345',
        '--dupe_factor', '5'])

task_runner = TaskRunner()
for task_id in range(8):
   task_runner.do_task(create_pretrain, task_id)
