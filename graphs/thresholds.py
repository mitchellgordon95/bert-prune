import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import re

color_map = {
    0: 'black',
    .1: 'grey',
    .2: 'darkblue',
    .3: 'blue',
    .4: 'darkgreen',
    .5: 'green',
    .6: 'yellow',
    .7: 'orange',
    .8: 'red',
    .9: 'purple',
}

def plot_thresholds(sparsity, ax):
    model_dir = f"models/pretrain/gradual_prune_{int(sparsity*100)}/"
    event_paths = sorted(glob.glob(os.path.join(model_dir, "event*")))

    # Just get the second last event of the last event file
    # (we only care about the last value of thresholds for now)
    event = list(tf.train.summary_iterator(event_paths[-1]))[-2]

    print(f'Training step {event.step}')

    x, y = [], []
    for value in event.summary.value:
        if 'threshold' in value.tag:
            tag = re.sub(r'model_pruning_summaries/bert/(.*)/threshold/threshold', r'\1', value.tag)
            if tag != 'embeddings/':
                x.append(value.simple_value)
                y.append(tag)
                # ax.scatter(value.simple_value, tag, color=color_map[sparsity])

    ax.plot(x, y, color=color_map[sparsity])

fig, ax = plt.subplots(figsize=(20,20))
for sparsity in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
    try:
        threshold = plot_thresholds(sparsity, ax)
    except IndexError:
        print(f'Missing gradual_prune_{int(sparsity*100)}')

# Add layer lines
for layer in range(12):
    y = f'encoder/layer_{layer}/output/fully_connected'
    ax.plot([0, .1], [y, y], '--', color='black')

ax.set_xlim(left=0)
plt.gcf().subplots_adjust(left=.2)
plt.savefig('tmp.png')
