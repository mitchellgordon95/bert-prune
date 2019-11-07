import tensorflow as tf
import os
import numpy as np
import fire
import matplotlib.pyplot as plt
from univa_grid import TaskRunner
from graphs.weight_change import sort_order_movements

def weight_change(first_model, second_model, filename):
    percentages = np.array([])
    with open(f'{filename}.txt', 'w+') as out_f:
        for var_name, _ in tf.train.list_variables(first_model):
            if var_name.endswith('/weights'):
                first_tensor = tf.contrib.framework.load_variable(first_model, var_name)
                second_tensor = tf.contrib.framework.load_variable(second_model, var_name)
                size = first_tensor.size
                movements = sort_order_movements(first_tensor, second_tensor)
                percentages = np.concatenate((percentages, movements / size))

        print(f'Average: {np.mean(percentages)}', file=out_f)
        print(f'Std: {np.std(percentages)}', file=out_f)

try:
    os.makedirs('graphs/weight_change/trained_downstream')
except:
    pass

if __name__ == '__main__':
    task_runner = TaskRunner()
    for task in ['CoLA', 'SST-2', 'QNLI', 'QQP', 'MNLI']: # 1-65 for uge
        for epoch in range(13):
            task_runner.do_task(
                weight_change,
                'models/pretrain/burned_in',
                f'models/{task}/trained_downstream_lr_5e-5_epoch_{epoch}' if epoch != 0 else 'models/pretrain/burned_in',
                f'graphs/weight_change/trained_downstream/{task}_{epoch}')
