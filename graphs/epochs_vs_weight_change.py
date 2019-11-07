from checkpoint_utils.diff_masks import diff_masks
import matplotlib.pyplot as plt
from tables.common import parse_file

plt.rcParams.update({'font.size': 12})

# Note: parse doesn't like scientific notation strings, so we manually convert to floats
DIFF_TEMPLATE = """Average: {avg}
Std: {std}
"""

fig, ax = plt.subplots()

for task in ['MNLI', 'QQP', 'QNLI', 'SST-2', 'CoLA']:
    print(task)
    lr = '5e-5'
    epochs = list(range(13))
    avgs = []
    stds = []
    for epoch in epochs:
        diff_results = parse_file(f'graphs/weight_change/trained_downstream/{task}_{epoch}.txt', DIFF_TEMPLATE)
        if not diff_results:
            print(f'Missing epoch {epoch}', end=' ')
            avgs.append(float('inf'))
            stds.append(0)
        else:
            avgs.append(float(diff_results['avg'])*100)
            stds.append(float(diff_results['std'])*100)

    ax.plot(epochs, avgs, label=task)

ax.set_title('Weight Sort Order Movement During Downstream Training')
ax.set_xlabel('Downstream Epochs Trained')
ax.set_ylabel('Avg % Movement of Weights in Sort Order')
ax.set_ylim(0)
ax.legend()
fig.savefig('epochs_vs_weight_change.png')
