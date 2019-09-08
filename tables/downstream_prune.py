from tables.common import grid_search_eval

table_rows = []
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
    eval_accs, train_losses = grid_search_eval(lambda task, lr: f'models/{task}/downstream_prune_{int(sparsity*100)}_lr_{lr}/eval_results.txt')
    avg_eval = sum(eval_accs) / len(eval_accs)
    avg_loss = sum(train_losses) / len(train_losses)
    table_rows.append(
        f"{sparsity} & ? & " + " & ".join([f"{acc:.2f}|{loss:.2f}" for acc, loss in zip(eval_accs,train_losses)]) + f" & {avg_eval:.2f}|{avg_loss:.2f}"
    )

rows = "\\\\\n".join(table_rows)
print(f"""
  &  & \\multicolumn{{7}}{{c}}{{Pruned after Downstream Fine-tuning}} \\\\
\\hline
{rows}\\\\
\\end{{tabular}}
""")
