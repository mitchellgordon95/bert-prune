import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile
import fire

def random_masks(model_dir, out_dir, sparsity: float):
    """Prunes a random [sparsity] of of weights in each matrix of [model_dir].
    Makes a new checkpoint [out_dir].
    """
    model_dir = model_dir.rstrip('/')

    with tf.Session() as sess:

        # Load all the variables from the checkpoint
        for var_name, _ in tf.train.list_variables(model_dir):
            tensor = tf.contrib.framework.load_variable(model_dir, var_name)

            if var_name.endswith('/mask'):
                num_zeros = int(tensor.size * sparsity)
                new_mask = np.concatenate((np.zeros(num_zeros), np.ones(tensor.size - num_zeros)))
                np.random.shuffle(new_mask)
                new_mask = new_mask.reshape(tensor.shape).astype(tensor.dtype)
                var = tf.Variable(new_mask, name=var_name)
            else:
                var = tf.Variable(tensor, name=var_name)

        # Save these new variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        os.mkdir(out_dir)
        saver.save(sess, os.path.join(out_dir, 'random_prune.ckpt'))

if __name__ == '__main__':
    fire.Fire(random_masks)
