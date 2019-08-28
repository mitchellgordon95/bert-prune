import tensorflow as tf
import numpy as np
import fire
import matplotlib.pyplot as plt
import os
from checkpoint_utils.dist_stats import cosort

def main(first_model, second_model, out_dir):
    os.makedirs(out_dir)
    for var_name, _ in tf.train.list_variables(first_model):
        if var_name.endswith('/weights'):
            first_tensor = tf.contrib.framework.load_variable(first_model, var_name)
            second_tensor = tf.contrib.framework.load_variable(second_model, var_name)

            y = np.sort(np.abs(first_tensor.flatten()))
            x = np.arange(first_tensor.size)
            fig, ax = plt.subplots()
            ax.scatter(x, y, s=1)
            var_name = var_name.replace('/', '_')
            fig.savefig(f'{out_dir}/{var_name}_before.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            y = cosort(first_tensor, second_tensor)
            ax.scatter(x, y, s=1)
            var_name = var_name.replace('/', '_')
            fig.savefig(f'{out_dir}/{var_name}_after.png')
            plt.close(fig)

if __name__ == '__main__':
    fire.Fire(main)
