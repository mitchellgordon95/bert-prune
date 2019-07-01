from parse import parse

EVAL_RESULTS_TEMPLATE = """eval_accuracy = {eval_accuracy:f}
eval_loss = {eval_loss:f}
global_step = {step:d}
loss = {loss:f}"""

PRETRAIN_EVAL_RESULTS_TEMPLATE = """global_step = {global_step:d}
loss = {loss:f}
masked_lm_accuracy = {masked_lm_accuracy:f}
masked_lm_loss = {masked_lm_loss:f}
next_sentence_accuracy = {next_sentence_accuracy:f}
next_sentence_loss = {next_sentence_loss:f}
"""

def parse_file(fname, template):
    try:
        with open(fname, 'r') as f:
            return parse(template, f.read())
    except FileNotFoundError:
        print(f"Missing {fname}")
        return None

table_rows = []
for sparsity in [0, .4, .5, .6, .7, .8, .9]:
    row_entries = []

    pretrain_results = parse_file(f'models/pretrain/gradual_prune_{int(sparsity*100)}/eval_results.txt', PRETRAIN_EVAL_RESULTS_TEMPLATE)
    if pretrain_results:
        row_entries.append(pretrain_results['loss'])
    else:
        row_entries.append(0)

    # Find the best accuracy for each task
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA', 'STS-B', 'MRPC', 'RTE']:

        # Select the model with the best accuracy among the grid search
        grid_search_acc = []
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            downstream_results = parse_file(f'models/{task}/gradual_prune_{int(sparsity*100)}_lr_{lr}/eval_results.txt', EVAL_RESULTS_TEMPLATE)
            if downstream_results:
                grid_search_acc.append(downstream_results['eval_accuracy'])

        row_entries.append(max(grid_search_acc, default=0))

    table_rows.append(
        f"{sparsity} & " + " & ".join([f"{acc:.2f}" for acc in row_entries])
    )

rows = "\\\\\n".join(table_rows)
print(f"""
\\begin{{tabular}}{{|c|c||c|c|c|c|c|c|c|c|}}
\\hline Pruned & Pre-train Loss & MNLI & QQP & QNLI & SST-2 & CoLA & STS-B & MRPC & RTE \\\\
\\hline
{rows}\\\\
\\end{{tabular}}
""")
