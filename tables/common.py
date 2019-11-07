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

def grid_search_eval(eval_path_fn):
    """Given eval_path_fn(task, lr), searches over some eval results and
    returns the best one for each task"""
    eval_entries, train_losses = [], []
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA']: # 'MRPC', 'RTE', 'STS-B'
        # Select the model with the best accuracy among the grid search
        grid_search_res = []
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            downstream_results = parse_file(eval_path_fn(task, lr), EVAL_RESULTS_TEMPLATE)
            train_eval = parse_file(eval_path_fn(task,lr)[:-16] + 'eval_train_results.txt', EVAL_RESULTS_TEMPLATE)
            train_loss = train_eval['loss'] if train_eval else float('inf')
            train_step = train_eval['step'] if train_eval else 0
            if downstream_results:
                grid_search_res.append((downstream_results['eval_accuracy'], train_loss, lr, downstream_results['step'], train_step))

        best = max(grid_search_res, default=(0,0,'',0, 0))
        # print(f'Using {task} with lr {best[2]} eval_step {best[3]} train_step {best[4]}')
        eval_entries.append(best[0])
        train_losses.append(best[1])

    return eval_entries, train_losses
