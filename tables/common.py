from parse import parse

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
    row_entries = []
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'MRPC', 'RTE']: # 'STS-B'

        # Select the model with the best accuracy among the grid search
        grid_search_acc = []
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            downstream_results = parse_file(eval_path_fn(task, lr), EVAL_RESULTS_TEMPLATE)
            if downstream_results:
                grid_search_acc.append(downstream_results['eval_accuracy'])

        row_entries.append(max(grid_search_acc, default=0))

    return row_entries
