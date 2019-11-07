from checkpoint_utils.diff_masks import diff_masks
import matplotlib.pyplot as plt
from tables.common import parse_file, EVAL_RESULTS_TEMPLATE
import numpy as np
plt.rcParams.update({'font.size': 14})

fig, eval_ax = plt.subplots()
diff_ax = eval_ax.twinx()

all_diffs = []
all_evals = []
for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA']:
    print(task)
    lr = '5e-5'
    epochs = [0,1,3,6,9,12]
    mask_diffs = []
    evals = []
    for epoch in epochs:
        model_name = f'models/{task}/trained_downstream_lr_{lr}_epoch_{epoch}_pruned_60_lr_{lr}'
        downstream_results = parse_file(f'{model_name}/eval_results.txt', EVAL_RESULTS_TEMPLATE)
        if downstream_results:
            evals.append(downstream_results['eval_accuracy'])
            mask_diff_res = parse_file(f'{model_name}/mask_diff.txt', '{diff:f}')
            if not mask_diff_res:
                mask_diff = diff_masks(model_name, f'models/{task}/trained_downstream_lr_{lr}_epoch_0_pruned_60_lr_{lr}')
                mask_diffs.append(mask_diff*100)
                with open(f'{model_name}/mask_diff.txt', 'w+') as out_f:
                    print(mask_diff, file=out_f)
            else:
                mask_diffs.append(mask_diff_res['diff']*100)
        else:
            print(f'Missing epoch {epoch}', end=' ')
            mask_diffs.append(float('inf'))
            evals.append(0)
    all_diffs.append(mask_diffs)
    all_evals.append(evals)

all_diffs = np.array(all_diffs).mean(axis=0)
all_evals = np.array(all_evals).mean(axis=0)

diff_ax.plot(epochs, all_diffs, color='red')
eval_ax.plot(epochs, all_evals, color='blue')

eval_ax.set_title('Effect of Fine-tuning Before Pruning 60%')
eval_ax.set_xlabel('Downstream Epochs Before Pruning')
eval_ax.set_ylabel('Avg GLUE Dev Acc', color='blue')
eval_ax.set_ylim((.65,1))
eval_ax.tick_params(axis='y', color='blue')
diff_ax.set_ylabel('Pruning Mask Difference (%)', color='red')
diff_ax.set_ylim((0, 6))
diff_ax.tick_params(axis='y', color='red')
fig.savefig('effects_finetuning.png')
