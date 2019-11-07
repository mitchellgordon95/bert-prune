from checkpoint_utils.diff_masks import diff_masks
import matplotlib.pyplot as plt
from tables.common import parse_file
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots()

for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA']: # 'STS-B'
    print()
    print(task)
    x = []
    y = []
    for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        print()
        print(f'SPARSITY {sparsity}')
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            mask_diff_res = parse_file(
                f'models/{task}/downstream_prune_{int(sparsity*100)}_lr_{lr}/mask_diff.txt',
                '{diff:f}')

            if not mask_diff_res:
                try:
                    y.append(
                        diff_masks(
                            f'models/{task}/downstream_prune_{int(sparsity*100)}_lr_{lr}',
                            f'models/pretrain/gradual_prune_{int(sparsity*100)}'
                            ) * 100)
                    x.append(sparsity)
                    with open(f'models/{task}/downstream_prune_{int(sparsity*100)}_lr_{lr}/mask_diff.txt', 'w+') as out_f:
                        print(y[-1], file=out_f)
                except FileNotFoundError:
                    print(f'Missing {lr}', end=' ')
            else:
                y.append(mask_diff_res['diff'])
                x.append(sparsity)


    ax.scatter(x, y, label=task)

ax.set_title('Pre-train vs. Downstream Pruning')
ax.set_xlabel('Sparsity')
ax.set_ylabel('Pruning Mask Difference (%)')
ax.legend()
fig.savefig('mask_differences.png')
