from parse import parse

EVAL_RESULTS_TEMPLATE = """eval_accuracy = {eval_accuracy:f}
eval_loss = {eval_loss:f}
global_step = {step:d}
loss = {loss:f}"""

table_rows = []
for sparsity in [0, .4, .5, .6, .7, .8, .9]:
    task_accuracies = []

    # Find the best accuracy for each task
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'MRPC', 'XNLI', 'RTE']:

        # Select the model with the best accuracy among the grid search
        grid_search_acc = []
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            fname = f'models/{task}/gradual_prune_{int(sparsity*100)}_lr_{lr}/eval_results.txt'
            try:
                with open(fname, 'r') as f:
                    acc = parse(EVAL_RESULTS_TEMPLATE, f.read())['eval_accuracy']
                    grid_search_acc.append(acc)
            except FileNotFoundError:
                print(f"Missing {fname}")

        task_accuracies.append(max(grid_search_acc, default=0))

    table_rows.append(
        f"{sparsity} & " + " & ".join([f"{acc:.2f}" for acc in task_accuracies])
    )

rows = "\\\\\n".join(table_rows)
print(f"""
\\begin{{tabular}}{{|c||c|c|c|c|c|c|c|c|}}
\\hline Pruned & MNLI & QQP & QNLI & SST-2 & CoLA & STS-B & MRPC & RTE \\\\
\\hline
{rows}\\\\
\\end{{tabular}}
""")
