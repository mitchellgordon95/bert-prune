from tables.common import parse_file, grid_search_eval

PRETRAIN_EVAL_RESULTS_TEMPLATE = """global_step = {global_step:d}
loss = {loss:f}
masked_lm_accuracy = {masked_lm_accuracy:f}
masked_lm_loss = {masked_lm_loss:f}
next_sentence_accuracy = {next_sentence_accuracy:f}
next_sentence_loss = {next_sentence_loss:f}
"""

table_rows = []
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:

    eval_entries, train_losses = grid_search_eval(lambda task, lr: f'models/{task}/gradual_prune_{int(sparsity*100)}_less_embed_lr_{lr}/eval_results.txt')

    pretrain_results = parse_file(f'models/pretrain/gradual_prune_0_head_pruned_{int(sparsity*100)}/eval_results.txt', PRETRAIN_EVAL_RESULTS_TEMPLATE)
    pretrain_loss = pretrain_results['loss'] if pretrain_results else float('inf')

    avg_eval = sum(eval_entries) / len(eval_entries)
    avg_loss = sum(train_losses) / len(train_losses)
    table_rows.append(
        f"{sparsity} & {pretrain_loss:.2f} & " #+ " & ".join([f"{acc:.2f}|{loss:.2f}" for acc, loss in zip(eval_entries, train_losses)]) + f" & {avg_eval:.2f}|{avg_loss:.2f}"
    )

rows = "\\\\\n".join(table_rows)
print(f"""
\\begin{{tabular}}{{ccccccccccc}}
Pruned & Pre-train Loss & MNLI & QQP & QNLI & SST-2 & CoLA & STS-B & MRPC & RTE & AVG\\\\
\hline
{rows}\\\\
\\end{{tabular}}
""")
