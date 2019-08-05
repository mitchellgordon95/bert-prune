import tensorflow as tf
import numpy as np
import re
import sys
import os
from shutil import copyfile

if len(sys.argv) != 2:
    print("Usage: permute_masks_test.py [pretrain-dir]")
    print("Runs some sanity checks on the [pretrain-dir]_randomized_mask")
    exit()
else:
    CHECKPOINT_DIR = sys.argv[1].rstrip('/')
    OUTPUT_DIR = CHECKPOINT_DIR + "_randomized_mask"

with tf.Session() as sess:

    # For each variable in the original checkpoint
    old_tensors = {}
    for var_name, _ in tf.train.list_variables(CHECKPOINT_DIR):
        if var_name.endswith('/mask'):
            old = tf.contrib.framework.load_variable(CHECKPOINT_DIR, var_name)
            new = tf.contrib.framework.load_variable(OUTPUT_DIR, var_name)

            assert np.array_equal(np.sum(new > 0), np.sum(old > 0))
            try:
                assert not np.array_equal(new, old)
            except AssertionError:
                print(f'{var_name} is the same in both old and new checkpoints.')
                print('This might be fine because it\'s "random," but too many of these is bad.')

    for var_name, _ in tf.train.list_variables(OUTPUT_DIR):
        assert var_name.endswith('/mask')
