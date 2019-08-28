import tensorflow as tf
import numpy as np
import fire
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation
import re

def anim_max(tensor, model_dir, var_name):
    tensor_abs = np.abs(tensor)
    fig = plt.figure()
    ax = plt.axes(xlim=(0, tensor.shape[0]), ylim=(0, tensor.shape[1]))
    ax.set_title('Matrix Weights in Descending Abs Magnitude Order')
    colors = ['red', 'green', 'blue', 'orange', 'violet', 'pink', 'grey', 'brown']

    def init():
        return tuple()

    def animate(i):
        ind = np.unravel_index(np.argmax(tensor_abs, axis=None), tensor_abs.shape)
        val = round(tensor_abs[ind], 2)
        tensor_abs[ind] = 0
        color_ind = ind[1] % len(colors)
        return ax.scatter([ind[0]], [ind[1]], color=colors[color_ind]), ax.set_title(val)

    anim = FuncAnimation(fig, animate, init_func=init,
                                frames=1000, interval=20, blit=True)

    anim.save(f'graphs/weights/{model_dir}/{var_name}_anim.mp4')
    plt.close(fig)

def abs_value_sort(tensor, model_dir, var_name):
    y = np.sort(np.abs(tensor.flatten()))
    x = np.arange(tensor.size)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=0)
    ax.set_ylim((0, 1))
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_sort.png')
    plt.close(fig)

def hist(tensor, model_dir, var_name):
    y = np.sort(np.abs(tensor.flatten()))
    fig, ax = plt.subplots()
    ax.hist(y, range=(0,1), bins=100)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_hist.png')
    plt.close(fig)

def plot_3d(tensor, model_dir, var_name):
    tensor_abs = np.abs(tensor)
    x, y = np.meshgrid(np.arange(tensor.shape[1]), np.arange(tensor.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, tensor_abs)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_3d.png')
    plt.close(fig)

def max_k(tensor, k, model_dir, var_name, reverse=False):
    tensor_abs = np.abs(tensor)
    argsort = np.argsort(tensor_abs, axis=None)
    if reverse:
        argsort = np.flip(argsort)
    x, y = np.unravel_index(argsort[-k:], tensor_abs.shape)
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=0)
    ax.set_title(f'Cutoff {tensor_abs[np.unravel_index(argsort[-k], tensor_abs.shape)]}')
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_{"min" if reverse else "max"}_{k}.png')
    plt.close(fig)

def heatmap(tensor, model_dir, var_name):
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(np.abs(tensor), vmax=0.1, cmap="YlGnBu", ax=ax)
    fig.savefig(f'graphs/weights/{model_dir}/{var_name}_heatmap.png')
    plt.close(fig)

def main(model_dir):
    os.makedirs(f'graphs/weights/{model_dir}')
    for var_name, _ in tf.train.list_variables(model_dir):
        if var_name.endswith('/weights') or var_name.endswith('/word_embeddings'):
            tensor = tf.contrib.framework.load_variable(model_dir, var_name)
            print(var_name)

            # Make the variable name more like a filename.
            var_name = var_name.replace('/', '_')
            # Make all layer numbers two digits wide so the filenames sort nicely
            var_name = re.sub(r'_(\d)_', r'_0\1_', var_name)

            max_k(tensor, 50000, model_dir, var_name)
            abs_value_sort(tensor, model_dir, var_name)
            hist(tensor, model_dir, var_name)
            heatmap(tensor, model_dir, var_name)


if __name__ == '__main__':
    fire.Fire(main)
