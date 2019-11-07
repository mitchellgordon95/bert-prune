from tables.common import grid_search_eval, parse_file
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
PRETRAIN_EVAL_RESULTS_TEMPLATE = """global_step = {global_step:d}
loss = {loss:f}
masked_lm_accuracy = {masked_lm_accuracy:f}
masked_lm_loss = {masked_lm_loss:f}
next_sentence_accuracy = {next_sentence_accuracy:f}
next_sentence_loss = {next_sentence_loss:f}
"""


def plot_avg_acc(eval_path_str, label, eval_ax, loss_ax):
    eval_avgs = []
    loss_avgs = []
    sparsities = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    for sparsity in sparsities:
        eval_accs, losses = grid_search_eval(lambda task, lr: eval_path_str.format(task=task,sparsity=int(sparsity*100),lr=lr))
        eval_avgs.append(sum(eval_accs) / len(eval_accs))
        loss_avgs.append(sum(losses) / len(losses))

    eval_ax.plot(sparsities, eval_avgs, label=label)
    print(loss_avgs)
    loss_ax.plot(sparsities, loss_avgs, label=label)

eval_fig, eval_ax = plt.subplots()
loss_fig, loss_ax = plt.subplots()

for label, eval_path_str in [
        ('prune pretrain','models/{task}/gradual_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        ('info deletion','models/{task}/gradual_prune_{sparsity}_no_mask_lr_{lr}/eval_results.txt'),
        ('prune downstream','models/{task}/downstream_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        ('random pruning','models/{task}/burned_in_random_prune_{sparsity}_lr_{lr}/eval_results.txt'),
        # ('sign ticket','models/{task}/gradual_prune_{sparsity}_sign_ticket_lr_{lr}/eval_results.txt'),
        # ('random mask reinit','models/{task}/gradual_prune_{sparsity}_randomized_mask_lr_{lr}/eval_results.txt'),
]:
    plot_avg_acc(eval_path_str, label, eval_ax, loss_ax)

eval_ax.set_title('Average GLUE Dev Acc')
eval_ax.set_xlabel('Prune Percentage')
eval_ax.set_ylabel('Dev Acc')
eval_ax.set_xlim(left=0)
eval_ax.set_ylim(bottom=0.65)
eval_ax.plot([0,1], [.872,.872], '--', label='BERT 0% Prune')
eval_ax.legend()
eval_fig.savefig('avg_glue_eval.png')

loss_ax.set_title('Average GLUE Training Loss')
loss_ax.set_xlabel('Prune Percentage')
loss_ax.set_ylabel('Train Loss')
loss_ax.set_xlim(left=0)
loss_ax.plot([0,1], [.16,.16], '--', label='BERT 0% Prune')
loss_ax.legend()
loss_fig.savefig('avg_glue_loss.png')

# Do pre-train vs. avg glue acc
eval_acc_list = []
eval_avgs = []
pretrain_losses = []
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
    eval_accs, _ = grid_search_eval(lambda task, lr: 'models/{task}/gradual_prune_{sparsity}_no_mask_lr_{lr}/eval_results.txt'.format(task=task,sparsity=int(sparsity*100),lr=lr))
    eval_acc_list.append(eval_accs)
    eval_avgs.append(sum(eval_accs) / len(eval_accs))
    pretrain_results = parse_file(f'models/pretrain/gradual_prune_{int(sparsity*100)}/eval_results.txt', PRETRAIN_EVAL_RESULTS_TEMPLATE)
    pretrain_losses.append(pretrain_results['loss'] if pretrain_results else float('inf'))

pretrain_fig, pretrain_ax = plt.subplots()
pretrain_ax.set_title('Pre-training Loss vs. Information Deletion Glue Accuracy')
pretrain_ax.set_xlabel('Pre-Training Loss')
pretrain_ax.set_ylabel('Glue Acc')
import numpy as np
eval_acc_np = np.array(eval_acc_list)
plt.gca().set_prop_cycle(None)
pretrain_ax.scatter(pretrain_losses, eval_acc_np[:,0], label="MNLI")
pretrain_ax.scatter(pretrain_losses, eval_acc_np[:,1], label="QQP")
pretrain_ax.scatter(pretrain_losses, eval_acc_np[:,2], label="QNLI")
pretrain_ax.scatter(pretrain_losses, eval_acc_np[:,3], label="SST-2")
pretrain_ax.scatter(pretrain_losses, eval_acc_np[:,4], label="CoLA")
plt.gca().set_prop_cycle(None)
for i in range(4):
    pretrain_ax.plot(pretrain_losses, np.poly1d(np.polyfit(pretrain_losses, eval_acc_np[:,i], 1))(pretrain_losses))
    print(f'{(eval_acc_np[-1,i] - eval_acc_np[0,i]) / (pretrain_losses[-1] - pretrain_losses[0])}')
# Exlude the last point it stands out.
pretrain_ax.plot(pretrain_losses[:-1], np.poly1d(np.polyfit(pretrain_losses[:-1], eval_acc_np[:-1,4], 1))(pretrain_losses[:-1]))
pretrain_ax.legend()
pretrain_fig.savefig('avg_glue_vs_pretrain.png')
