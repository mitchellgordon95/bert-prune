from tables.common import parse_file, EVAL_RESULTS_TEMPLATE
import matplotlib.pyplot as plt
import numpy as np


for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
    fig, ax = plt.subplots()
    for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA']:
        evals = []
        mask_diffs = []
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            downstream_results = parse_file(f'models/{task}/downstream_prune_{int(100*sparsity)}_lr_{lr}/eval_results.txt', EVAL_RESULTS_TEMPLATE)
            if downstream_results:
                evals.append(downstream_results['eval_accuracy'])
                mask_diffs.append(parse_file(f'models/{task}/downstream_prune_{int(100*sparsity)}_lr_{lr}/mask_diff.txt', '{diff:f}')['diff'])
            else:
                print(f'Missing {task} {sparsity} {lr}')
        mask_diffs = np.array(mask_diffs)
        evals = np.array(evals)
        ax.plot(np.sort(mask_diffs), evals[np.argsort(mask_diffs)], label=f'{task}')

    ax.set_title('Mask Diff vs. Eval Acc')
    ax.set_xlabel('Mask Diff')
    ax.set_ylabel('Dev Acc')
    ax.legend()
    fig.savefig(f'mask_diff_eval_{sparsity}.png')

