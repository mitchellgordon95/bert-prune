from tables.common import grid_search_eval

table_rows = []
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
    row_entries = grid_search_eval(lambda task, lr: f'models/{task}/downstream_prune_{int(sparsity*100)}_lr_{lr}/eval_results.txt')
    avg = sum(row_entries) / len(row_entries)
    table_rows.append(
        f"{sparsity} & ? & " + " & ".join([f"{acc:.2f}" for acc in row_entries]) + f" & {avg:.2f}"
    )

rows = "\\\\\n".join(table_rows)
print(f"""
  &  & \\multicolumn{{7}}{{c}}{{Pruned after Downstream Fine-tuning}} \\\\
\\hline
{rows}\\\\
\\end{{tabular}}
""")
