from tables.common import grid_search_eval
import matplotlib.pyplot as plt


def plot_avg_acc(eval_path_str, label, ax):
    avgs = []
    sparsities = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    for sparsity in sparsities:
        rows_vals, _ = grid_search_eval(lambda task, lr: eval_path_str.format(task=task,sparsity=int(sparsity*100),lr=lr))
        avgs.append(sum(rows_vals) / len(rows_vals))

    ax.plot(sparsities, avgs, label=label)

fig, ax = plt.subplots()

for label, eval_path_str in [
        ('prune pretrain','models/{task}/gradual_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        ('prune downstream','models/{task}/downstream_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        ('prune then no mask','models/{task}/gradual_prune_{sparsity}_no_mask_lr_{lr}/eval_results.txt'),
        ('sign ticket','models/{task}/gradual_prune_{sparsity}_sign_ticket_lr_{lr}/eval_results.txt'),
]:
    plot_avg_acc(eval_path_str, label, ax)

ax.set_xlim(left=0)
ax.legend()
plt.savefig('tmp2.png')
