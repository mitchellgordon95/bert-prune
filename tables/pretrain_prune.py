from tables.common import parse_file, grid_search_eval

EVAL_RESULTS_TEMPLATE = """eval_accuracy = {eval_accuracy:f}
eval_loss = {eval_loss:f}
global_step = {step:d}
loss = {loss:f}"""

table_rows = []
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:

    row_entries = grid_search_eval(lambda task, lr: f'models/{task}/gradual_prune_{int(sparsity*100)}_lr_{lr}/eval_results.txt')

    pretrain_results = parse_file(f'models/pretrain/gradual_prune_{int(sparsity*100)}/eval_results.txt', PRETRAIN_EVAL_RESULTS_TEMPLATE)
    if pretrain_results:
        row_entries.append(pretrain_results['loss'])
    else:
        row_entries.append(0)

    avg = sum(row_entries[1:]) / (len(row_entries) - 1)
    table_rows.append(
        f"{sparsity} & " + " & ".join([f"{acc:.2f}" for acc in row_entries]) + f" & {avg:.2f}"
    )

rows = "\\\\\n".join(table_rows)
print(f"""
\\begin{{tabular}}{{ccccccccccc}}
Pruned & Pre-train Loss & MNLI & QQP & QNLI & SST-2 & CoLA & STS-B & MRPC & RTE & AVG\\\\
\hline
{rows}\\\\
\\end{{tabular}}
""")
