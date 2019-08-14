from parse import parse
import tensorflow as tf
import glob
import os

EVAL_RESULTS_TEMPLATE = """eval_accuracy = {eval_accuracy:f}
eval_loss = {eval_loss:f}
global_step = {step:d}
loss = {loss:f}"""

def parse_file(fname, template):
    try:
        with open(fname, 'r') as f:
            return parse(template, f.read())
    except FileNotFoundError:
        print(f"Missing {fname}")
        return None

def last_training_loss(model_dir):
    event_paths = sorted(glob.glob(os.path.join(model_dir, "event*")))

    for event_path in event_paths:
        for event in tf.train.summary_iterator(event_path):
            for value in event.summary.value:
                if value.tag == 'loss_1':
                    last_loss = value.simple_value
    return last_loss

def grid_search_eval(eval_path_fn):
    """Given eval_path_fn(task, lr), searches over some eval results and
    returns the best one for each task"""
    eval_entries, train_losses = [], []
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'MRPC', 'RTE']: # 'STS-B'

        # Select the model with the best accuracy among the grid search
        grid_search_res = []
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            downstream_results = parse_file(eval_path_fn(task, lr), EVAL_RESULTS_TEMPLATE)
            train_loss = last_training_loss(eval_path_fn(task,lr)[:-16])
            if downstream_results:
                grid_search_res.append((downstream_results['eval_accuracy'], train_loss))

        best = max(grid_search_res, default=(0,0))
        eval_entries.append(best[0])
        train_losses.append(best[1])

    return eval_entries, train_losses
