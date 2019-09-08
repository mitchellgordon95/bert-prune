import tensorflow as tf
import numpy as np
import fire
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import re
from collections import Counter

def anim_max(tensor, model_dir, var_name):
    pass # Ressurectable from git history

def abs_value_sort(tensor, model_dir, var_name):
    y = np.sort(np.abs(tensor.flatten()))
    x = np.arange(tensor.size)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1)
    ax.set_ylim((0, 1))
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_sort.png')
    plt.close(fig)

def hist(tensor, model_dir, var_name):
    y = np.abs(tensor.flatten())
    fig, ax = plt.subplots()
    ax.hist(y, density=True, range=(0,0.5), bins=100)
    ax.set_ylim((0, 25))
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_hist.png')
    plt.close(fig)

    y = tensor.flatten()
    fig, ax = plt.subplots()
    ax.set_ylim((0, 12))
    ax.hist(y, density=True, range=(-0.5,0.5), bins=100)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_hist_signed.png')
    plt.close(fig)

def plot_3d(tensor, model_dir, var_name):
    pass # Ressurectable from git history

def max_k(tensor, k, model_dir, var_name, reverse=False):
    pass # Ressurectable from git history

def heatmap(tensor, model_dir, var_name):
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(np.abs(tensor), vmax=0.1, cmap="YlGnBu", ax=ax)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_heatmap.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(tensor, vmin=-0.1, vmax=0.1, cmap="YlGnBu", ax=ax)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_heatmap_signed.png')
    plt.close(fig)

def attention_head_hist(ledger, model_dir, layers=12, heads=12, sparsity=0.5):
    print(f"Warning: assuming BERT has {layers} layers and {heads} attention heads")
    data_points = []
    for layer in range(layers):
        var_names = [var_name for var_name in ledger.keys() if f'layer_{layer}/attention' in var_name]

        # There should be the key, the query, the value, and the output matrices
        assert len(var_names) == 4

        # TODO: attn heads are columns in attn matrices, rows in FC, I think...
        head_size = Counter()
        head_pruned_count = Counter()

        for var_name in var_names:
            print(var_name)

            tensor = np.abs(ledger[var_name])
            mask = prune(tensor, sparsity)
            head_width = tensor.shape[0] // 12 if 'output' not in var_name else tensor.shape[1] // 12
            assert head_width == tensor.shape[0] / 12 if 'output' not in var_name else tensor.shape[1] / 12
            for head in range(heads):
                if 'output' not in var_name:
                    head_mask = mask[head_width*head:head_width*(head+1)][:]
                else:
                    head_mask = mask[:][head_width*head:head_width*(head+1)]

                head_size[head] += head_mask.size
                head_pruned_count[head] += np.sum(head_mask)

        for head in range(heads):
            data_points.append(head_pruned_count[head] / head_size[head])

    fig, ax = plt.subplots()
    ax.hist(data_points, range=(0,1), bins=100)
    fig.savefig(f'graphs/weights/{model_dir}/attn_head_pruning_{sparsity}.png')
    plt.close(fig)

def pruned_weights_sum(ledger, model_dir):
    sparsities = np.linspace(0, 1, endpoint=False)
    y1 = []
    y2 = []
    for sparsity in sparsities:
        sum_abs = 0
        abs_sum = 0
        for var_name, tensor in ledger.items():
            mask = prune(tensor, sparsity)
            sum_abs += np.sum(np.abs(tensor[mask]))
            abs_sum += np.abs(np.sum(tensor[mask]))

        y1.append(sum_abs)
        y2.append(abs_sum)

    fig, ax = plt.subplots()
    ax.plot(sparsities, y1, label='Sum of Magnitudes')
    ax.plot(sparsities, y2, label='Magnitude of Sum')
    ax.legend()
    fig.savefig(f'graphs/weights/{model_dir}/sum_weights_pruned.png')
    plt.close(fig)


def prune(tensor, sparsity):
    """Returns the mask that would be used to prune tensor to the specified sparsity"""
    tensor = np.abs(tensor)
    thresh_ind = int(tensor.size * sparsity)
    threshold = np.partition(tensor.flatten(), thresh_ind)[thresh_ind]
    return tensor < threshold

ledger = {}

def main(model_dir):
    os.makedirs(f'graphs/weights/{model_dir}')
    for var_name, _ in tf.train.list_variables(model_dir):
        if var_name.endswith('/weights') or var_name.endswith('/word_embeddings'):
            tensor = tf.contrib.framework.load_variable(model_dir, var_name)
            ledger[var_name] = tensor
            print(var_name)

            # Make the variable name more like a filename.
            var_name = var_name.replace('/', '_')
            # Make all layer numbers two digits wide so the filenames sort nicely
            var_name = re.sub(r'_(\d)_', r'_0\1_', var_name)

            hist(tensor, model_dir, var_name)
            heatmap(tensor, model_dir, var_name)

    for sparsity in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        attention_head_hist(ledger, model_dir)
    pruned_weights_sum(ledger, model_dir)

if __name__ == '__main__':
    fire.Fire(main)
