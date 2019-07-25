import tensorflow as tf
import numpy as np
import fire


def diff_masks(first_model, second_model):
    """Prints the percentage of mask values that are different between the first and second model"""

    # Load all the variables from the checkpoint
    total_masks = 0
    diff_masks = 0
    for var_name, _ in tf.train.list_variables(first_model):
        if var_name.endswith('/mask'):
            first_tensor = tf.contrib.framework.load_variable(first_model, var_name)
            second_tensor = tf.contrib.framework.load_variable(second_model, var_name)

            total_masks += first_tensor.size
            diff_masks += np.sum(first_tensor != second_tensor)

    return diff_masks / total_masks


if __name__ == '__main__':
    fire.Fire(diff_masks)
