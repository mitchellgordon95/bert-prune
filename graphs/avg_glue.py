from tables.common import grid_search_eval
import matplotlib.pyplot as plt


def plot_avg_acc(eval_path_str, label, eval_ax, loss_ax):
    eval_avgs = []
    loss_avgs = []
    sparsities = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    for sparsity in sparsities:
        eval_accs, losses = grid_search_eval(lambda task, lr: eval_path_str.format(task=task,sparsity=int(sparsity*100),lr=lr))
        eval_avgs.append(sum(eval_accs) / len(eval_accs))
        loss_avgs.append(sum(losses) / len(losses))

    eval_ax.plot(sparsities, eval_avgs, label=label)
    loss_ax.plot(sparsities, loss_avgs, label=label)

eval_fig, eval_ax = plt.subplots()
loss_fig, loss_ax = plt.subplots()

for label, eval_path_str in [
        ('prune pretrain','models/{task}/gradual_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        ('prune downstream','models/{task}/downstream_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        ('prune then no mask','models/{task}/gradual_prune_{sparsity}_no_mask_lr_{lr}/eval_results.txt'),
        # ('sign ticket','models/{task}/gradual_prune_{sparsity}_sign_ticket_lr_{lr}/eval_results.txt'),
        # ('random mask reinit','models/{task}/gradual_prune_{sparsity}_randomized_mask_lr_{lr}/eval_results.txt'),
]:
    plot_avg_acc(eval_path_str, label, eval_ax, loss_ax)

eval_ax.set_title('Average GLUE Dev Acc')
eval_ax.set_xlabel('Prune Percentage')
eval_ax.set_ylabel('Dev Acc')
eval_ax.set_xlim(left=0)
eval_ax.set_ylim(bottom=0.5)
eval_ax.plot([0,1], [.84,.84], '--', label='BERT 0% Prune')
eval_ax.legend()
eval_fig.savefig('avg_glue_eval.png')

loss_ax.set_title('Average GLUE Training Loss')
loss_ax.set_xlabel('Prune Percentage')
loss_ax.set_ylabel('Train Loss')
loss_ax.set_xlim(left=0)
loss_ax.plot([0,1], [.19,.19], '--', label='BERT 0% Prune')
loss_ax.legend()
loss_fig.savefig('avg_glue_loss.png')
