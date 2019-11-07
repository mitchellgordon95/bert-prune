import tensorflow as tf
import os
import numpy as np
import fire
import matplotlib.pyplot as plt
from univa_grid import TaskRunner


def sort_order_movements(x, y):
    y_xsorted = cosort(x,y)
    y_argsort = np.argsort(y_xsorted)
    return np.abs(y_argsort - np.arange(y_argsort.size))

def cosort(x, y):
    """Sort x, and apply the same swapping operations to y. Return y."""
    assert x.shape == y.shape
    x = np.abs(x.flatten())
    y = np.abs(y.flatten())

    # If we sorted x, this is the order we would look at the elements
    x_argsort = np.argsort(x)

    # Now, take the elements of y in that order
    return y[x_argsort]

def weight_change(first_model, second_model, filename):
    BIN_SIZE = 100
    heat_map = np.zeros(BIN_SIZE)
    percentages = np.array([])
    with open(f'{filename}.txt', 'w+') as out_f:
        for var_name, _ in tf.train.list_variables(first_model):
            if var_name.endswith('/weights'):
                first_tensor = tf.contrib.framework.load_variable(first_model, var_name)
                second_tensor = tf.contrib.framework.load_variable(second_model, var_name)

                size = first_tensor.size
                movements = sort_order_movements(first_tensor, second_tensor)
                avg_movement = np.mean(movements)
                std_movement = np.std(movements)
                print(f'{avg_movement:.0f} +- {std_movement:.0f} / {size} = {avg_movement / size:.3f} +- {std_movement / size:.3f}', file=out_f)
                print(f'{avg_movement:.0f} +- {std_movement:.0f} / {size} = {avg_movement / size:.3f} +- {std_movement / size:.3f}')

                percentages = np.concatenate((percentages, movements / size))

                bin_width = int(size / BIN_SIZE)
                for i in range(BIN_SIZE):
                    heat_map[i] += np.sum(movements[i*bin_width:(i+1)*bin_width])

        print(f'Average: {np.mean(percentages)}', file=out_f)
        print(f'Std: {np.std(percentages)}', file=out_f)

    fig, ax = plt.subplots()
    ax.set_title('Distribution of Sort Order Movements')
    ax.set_xlabel('Starting Magnitude Sort Order Position')
    ax.set_ylabel('% of Movement')
    ax.bar(range(BIN_SIZE), heat_map / np.sum(heat_map))
    plt.savefig(filename)
    # return total_inversions / possible_inversions

try:
    os.makedirs('graphs/weight_change')
except:
    pass

if __name__ == '__main__':
    task_runner = TaskRunner()
    for task in ['CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']: # 1 -20 for uge
        for lr in ['2e-5','3e-5','4e-5','5e-5']:
            task_runner.do_task(
                weight_change,
                'models/pretrain/gradual_prune_0',
                f'models/{task}/gradual_prune_0_lr_{lr}',
                f'graphs/weight_change/{task}_{lr}')
