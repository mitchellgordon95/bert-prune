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
from checkpoint_utils.common import prune
from checkpoint_utils.prune_attn_heads import params_for_attn, attn_head_weight

plt.rcParams.update({'font.size': 12})

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
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(np.abs(tensor), vmax=0.1, cmap="YlGnBu", ax=ax)
    ax.set_title('Parameter Matrix Magnitude Heatmap', fontsize=20)
    ax.set_ylabel('Row', fontsize=20)
    ax.set_xlabel('Column', fontsize=20)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_heatmap.png')
    plt.close(fig)

    # fig, ax = plt.subplots(figsize=(20,20))
    # sns.heatmap(tensor, vmin=-0.1, vmax=0.1, cmap="YlGnBu", ax=ax)
    # plt.tight_layout()
    # fig.savefig(f'graphs/weights/{model_dir}/{var_name}_heatmap_signed.png')
    # plt.close(fig)

def attention_head_plot(ledger, model_dir, layers=12, heads=12):
    print(f"Warning: assuming BERT has {layers} layers and {heads} attention heads")
    fig, ax = plt.subplots()
    sparsities = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    y_avg = []
    y_min = []
    y_max = []
    for sparsity in sparsities:
        print(sparsity)

        prune_percents = np.zeros((12, 12)) # 12 layers, 12 heads each
        for layer in range(12):
            parameter_masks = tuple(prune(mat, sparsity) for mat in params_for_attn(ledger, layer))
            ones = tuple(np.ones_like(mat) for mat in parameter_masks)

            for head in range(12):
                not_pruned = attn_head_weight(*parameter_masks, head)
                total = attn_head_weight(*ones, head)
                prune_percents[layer, head] = (total - not_pruned) / total

        prune_percents = prune_percents.flatten()
        fig2, ax2 = plt.subplots()
        ax2.hist(prune_percents, range=(0,1), bins=100)
        fig2.savefig(f'graphs/weights/{model_dir}/attn_head_pruning_{sparsity}.png')
        plt.close(fig2)
        y_avg.append(np.mean(prune_percents))
        y_min.append(np.min(prune_percents))
        y_max.append(np.max(prune_percents))

    # The size of the error bars, not the min and max
    y_err = np.array([y_min, y_max])
    y_err = np.abs(y_err - y_avg)
    ax.errorbar(sparsities, y_avg, yerr=y_err)
    ax.set_title('% of Individual Attention Head Pruned')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('% of Attn Head Pruned')
    fig.savefig(f'graphs/weights/{model_dir}/attn_head_pruning.png')
    plt.close(fig)

def pruned_weights_sum(ledger, model_dir):
    sparsities = np.linspace(0, 1, endpoint=False)
    y1 = []
    y2 = []
    for sparsity in sparsities:
        sum_abs = 0
        sum_ = 0
        for var_name, tensor in ledger.items():
            pruned = prune(tensor, sparsity) == 0
            sum_abs += np.sum(np.abs(tensor[mask]))
            sum_ += np.sum(tensor[mask])

        y1.append(sum_abs)
        y2.append(sum_)

    fig, ax = plt.subplots()
    ax.plot(sparsities, y1, label='Sum of Absolute')
    ax.plot(sparsities, y2, label='Sum of Signed')
    ax.set_title('Sum of Weights Pruned')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Sum')
    # ax.set_ylim((-60000, 60000))
    ax.plot([0, 1], [0, 0], '--', label='y=0')
    ax.legend()
    fig.savefig(f'graphs/weights/{model_dir}/sum_weights_pruned.png')
    plt.close(fig)

ledger = {}

def main(model_dir):
    try:
        os.makedirs(f'graphs/weights/{model_dir}')
    except FileExistsError:
        pass
    with open(f'graphs/weights/{model_dir}/normals.txt', 'w+') as normals_f:
        for var_name, _ in tf.train.list_variables(model_dir):
            if var_name.endswith('/weights') or var_name.endswith('/mask'):# or var_name.endswith('/word_embeddings'):
                tensor = tf.contrib.framework.load_variable(model_dir, var_name)
                ledger[var_name] = tensor
                print(var_name)

                # print(var_name, end=' & ', file=normals_f)
                # print(f'{np.mean(tensor):.4f}', end=' & ', file=normals_f)
                # print(f'{np.std(tensor):.3f}', file=normals_f)

                # Make the variable name more like a filename.
                var_name = var_name.replace('/', '_')
                # Make all layer numbers two digits wide so the filenames sort nicely
                var_name = re.sub(r'_(\d)_', r'_0\1_', var_name)


                # hist(tensor, model_dir, var_name)
                # heatmap(tensor, model_dir, var_name)

    attention_head_plot(ledger, model_dir)
    # pruned_weights_sum(ledger, model_dir)

if __name__ == '__main__':
    fire.Fire(main)
